using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// GT-CRN-SE speech separation implementation
    /// </summary>
    internal class SepProjOfGtcrnSe : ISepProj, IDisposable
    {
        #region Constants
        private const int StftWinLen = 512;       // STFT window length
        private const int StftFftLen = 512;       // FFT length
        private const string StftWinType = "hanning";  // STFT window type
        private const int StftWinInc = 256;       // STFT window increment
        private const float TrimDuration = 0.1f;  // Audio tail trimming duration (seconds)
        #endregion

        #region Private Fields
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength;
        private int _shiftLength;
        #endregion

        #region Constructor
        /// <summary>
        /// Initializes a new instance of the SepProjOfGtcrnSe class
        /// </summary>
        /// <param name="sepModel">Separation model wrapper</param>
        /// <exception cref="ArgumentNullException">Thrown when sepModel is null</exception>
        public SepProjOfGtcrnSe(SepModel sepModel)
        {
            _modelSession = sepModel?.ModelSession ?? throw new ArgumentNullException(nameof(sepModel), "Separation model cannot be null");
            _customMetadata = sepModel.CustomMetadata;
            _featureDim = sepModel.FeatureDimension;
            _sampleRate = sepModel.SampleRate;
            _channels = sepModel.Channels;
            _chunkLength = sepModel.ChunkLength;
            _shiftLength = sepModel.ShiftLength;
        }
        #endregion

        #region Public Properties
        public InferenceSession ModelSession
        {
            get => _modelSession;
            set => _modelSession = value ?? throw new ArgumentNullException(nameof(value), "Model session cannot be null");
        }

        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Channels { get => _channels; set => _channels = value; }
        #endregion

        #region State Handling Methods
        /// <summary>
        /// Stacks a list of state lists into a single state list
        /// </summary>
        /// <param name="statesList">List of state lists to stack</param>
        /// <returns>Stacked state array</returns>
        /// <exception cref="ArgumentNullException">Thrown when statesList is null or empty</exception>
        public List<float[]> StackStates(List<List<float[]>> statesList)
        {
            if (statesList == null || statesList.Count == 0)
                throw new ArgumentNullException(nameof(statesList), "States list collection cannot be null or empty");

            return statesList[0];
        }

        /// <summary>
        /// Unstacks a single state list into a list of state lists
        /// </summary>
        /// <param name="states">State array to unstack</param>
        /// <returns>Unstacked list of state lists</returns>
        /// <exception cref="ArgumentNullException">Thrown when states is null</exception>
        public List<List<float[]>> UnstackStates(List<float[]> states)
        {
            if (states == null)
                throw new ArgumentNullException(nameof(states), "States array cannot be null");

            Debug.Assert(states.Count % 2 == 0, "When stacking states, state_list[0] length must be even");
            return new List<List<float[]>> { states };
        }
        #endregion

        #region Spectrogram Conversion Methods
        /// <summary>
        /// Converts a complex spectrogram array to STFT format (real + imaginary parts)
        /// </summary>
        /// <param name="complexArray">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [freq_bins, time_frames, 2]</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexArray)
        {
            int freqBins = complexArray.GetLength(0);    // Frequency bins (e.g., 961)
            int timeFrames = complexArray.GetLength(2);  // Time frames (e.g., 1808)

            // Target shape: [freq_bins, time_frames, 2] (real part + imaginary part)
            float[,,] stftArray = new float[freqBins, timeFrames, 2];

            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    stftArray[f, t, 0] = (float)complexArray[f, 0, t].Real;      // Real part
                    stftArray[f, t, 1] = (float)complexArray[f, 0, t].Imaginary; // Imaginary part
                }
            }

            return stftArray;
        }

        /// <summary>
        /// Converts back from STFT format to complex spectrogram
        /// </summary>
        /// <param name="stftFormat">STFT format array with shape [freq_bins, time_frames, 2]</param>
        /// <returns>Complex spectrogram with shape [freq_bins, 1, time_frames]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);    // Frequency bins (e.g., 257)
            int timeFrames = stftFormat.GetLength(1);  // Time frames (e.g., 609)

            // Create target complex array with middle dimension as 1
            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            // Iterate through each frequency and time point
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    float real = stftFormat[f, t, 0];   // Real part from last dimension
                    float imag = stftFormat[f, t, 1];   // Imaginary part from last dimension

                    // Create complex number (middle dimension index is 0)
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }
            }

            return complexSpectrogram;
        }
        #endregion

        #region Model Processing Methods
        /// <summary>
        /// Performs model separation projection
        /// </summary>
        /// <param name="modelInputs">List of model input entities</param>
        /// <param name="statesList">List of states (optional)</param>
        /// <param name="offset">Offset value (optional)</param>
        /// <returns>List of separated audio output entities</returns>
        /// <exception cref="ArgumentNullException">Thrown when modelInputs is null or empty</exception>
        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            if (modelInputs == null || modelInputs.Count == 0)
                throw new ArgumentNullException(nameof(modelInputs), "Model input list cannot be null or empty");

            try
            {
                int batchSize = modelInputs.Count;
                long[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
                float[] samples = PadHelper.PadSequence(modelInputs);

                // Process STFT and model inference
                var stftArgs = CreateStftArgs();
                Complex[,,] stftComplex = AudioProcessing.ComputeStft(samples, stftArgs, normalized: false);
                float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
                float[] flattenedSpectrum = spectrum.Cast<float>().ToArray();

                Tensor<float> outputTensor = RunInference(flattenedSpectrum, batchSize);
                if (outputTensor == null)
                    return new List<ModelOutputEntity>();

                return ProcessInferenceOutput(outputTensor, stftArgs, samples, modelInputs[0]);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Model projection failed: {ex.Message}\nStack trace: {ex.StackTrace}");
                throw; // Re-throw to prevent silent failure
            }
        }

        /// <summary>
        /// Generator projection (not implemented)
        /// </summary>
        /// <param name="modelOutputEntity">Model output entity</param>
        /// <param name="batchSize">Batch size</param>
        /// <returns>Empty list</returns>
        public List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1)
        {
            return new List<ModelOutputEntity>();
        }
        #endregion

        #region Private Helper Methods
        /// <summary>
        /// Creates STFT configuration arguments
        /// </summary>
        /// <returns>STFT configuration object</returns>
        private Utils.StftParameters CreateStftArgs()
        {
            return new Utils.StftParameters
            {
                WindowLength = StftWinLen,
                FftLength = StftFftLen,
                WindowType = StftWinType,
                WindowIncrement = StftWinInc
            };
        }

        /// <summary>
        /// Runs ONNX model inference
        /// </summary>
        /// <param name="inputData">Flattened input spectrum data</param>
        /// <param name="batchSize">Batch size</param>
        /// <returns>Inference output tensor</returns>
        /// <exception cref="InvalidOperationException">Thrown when model session is uninitialized</exception>
        private Tensor<float> RunInference(float[] inputData, int batchSize)
        {
            if (_modelSession == null)
                throw new InvalidOperationException("Model session is not initialized");

            var inputMeta = _modelSession.InputMetadata;
            var inputContainer = new List<NamedOnnxValue>();

            foreach (var inputName in inputMeta.Keys)
            {
                if (inputName == "input")
                {
                    int frameCount = inputData.Length / batchSize / 257 / 2;
                    int[] tensorShape = { batchSize, 257, frameCount, 2 };
                    var inputTensor = new DenseTensor<float>(inputData, tensorShape, false);
                    inputContainer.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
                }
            }

            using (var inferenceResults = _modelSession.Run(inputContainer))
            {
                return inferenceResults?.FirstOrDefault()?.AsTensor<float>();
            }
        }

        /// <summary>
        /// Processes inference output to generate final audio
        /// </summary>
        /// <param name="outputTensor">Inference output tensor</param>
        /// <param name="stftArgs">STFT configuration</param>
        /// <param name="originalSamples">Original input samples</param>
        /// <param name="inputEntity">First input entity for metadata</param>
        /// <returns>List of model output entities</returns>
        private List<ModelOutputEntity> ProcessInferenceOutput(
            Tensor<float> outputTensor,
            Utils.StftParameters stftArgs,
            float[] originalSamples,
            ModelInputEntity inputEntity)
        {
            float[,,] outputSpectrum = To3DArray(outputTensor);
            Complex[,,] complexSpectrum = ConvertSTFTFormatToComplex(outputSpectrum);
            float[] rawOutput = AudioProcessing.ComputeIstft(complexSpectrum, stftArgs, originalSamples.Length, normalized: false);
            float[] trimmedOutput = TrimAudio(rawOutput, inputEntity.SampleRate, inputEntity.Channels);

            return new List<ModelOutputEntity>
            {
                new ModelOutputEntity
                {
                    StemName = "vocals",
                    StemContents = trimmedOutput
                }
            };
        }

        /// <summary>
        /// Trims excess samples from audio tail
        /// </summary>
        /// <param name="audio">Audio data to trim</param>
        /// <param name="sampleRate">Sample rate of audio</param>
        /// <param name="channels">Number of audio channels</param>
        /// <returns>Trimmed audio data</returns>
        private float[] TrimAudio(float[] audio, int sampleRate, int channels)
        {
            int trimSampleCount = (int)(sampleRate * channels * TrimDuration);
            int outputLength = Math.Max(0, audio.Length - trimSampleCount);

            float[] trimmed = new float[outputLength];
            Array.Copy(audio, 0, trimmed, 0, outputLength);

            return trimmed;
        }

        /// <summary>
        /// Converts a tensor to a 3D float array
        /// </summary>
        /// <param name="tensor">Input tensor (3D or 4D)</param>
        /// <returns>3D float array</returns>
        /// <exception cref="ArgumentException">Thrown when tensor is not 3D or 4D, or 4D tensor's first dimension is not 1</exception>
        public float[,,] To3DArray(Tensor<float> tensor)
        {
            int[] dimensions = tensor.Dimensions.ToArray();

            // Validate tensor rank
            if (tensor.Rank != 3 && tensor.Rank != 4)
                throw new ArgumentException("Tensor must be 3-dimensional or 4-dimensional");

            // Handle 4D tensor (assume first dimension is 1)
            if (tensor.Rank == 4)
            {
                if (dimensions[0] != 1)
                    throw new ArgumentException("4-dimensional tensor must have first dimension equal to 1");

                dimensions = new[] { dimensions[1], dimensions[2], dimensions[3] };
            }

            float[,,] result = new float[dimensions[0], dimensions[1], dimensions[2]];
            int[] indices = new int[tensor.Rank];

            // Populate 3D array from tensor
            for (indices[0] = 0; indices[0] < dimensions[0]; indices[0]++)
            {
                for (indices[1] = 0; indices[1] < dimensions[1]; indices[1]++)
                {
                    for (indices[2] = 0; indices[2] < dimensions[2]; indices[2]++)
                    {
                        result[indices[0], indices[1], indices[2]] = tensor.Rank == 3
                            ? tensor[indices[0], indices[1], indices[2]]
                            : tensor[0, indices[0], indices[1], indices[2]];
                    }
                }
            }

            return result;
        }
        #endregion

        #region IDisposable Implementation
        /// <summary>
        /// Releases resources used by the instance
        /// </summary>
        /// <param name="disposing">True to release managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Release managed resources
                    _modelSession?.Dispose();
                    _modelSession = null;
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer (releases unmanaged resources)
        /// </summary>
        ~SepProjOfGtcrnSe()
        {
            Dispose(disposing: false);
        }

        /// <summary>
        /// Releases all resources used by the instance
        /// </summary>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}