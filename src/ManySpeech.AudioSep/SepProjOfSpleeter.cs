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
    /// Spleeter-based speech separation implementation
    /// </summary>
    internal class SepProjOfSpleeter : ISepProj, IDisposable
    {
        #region Constants
        private const int FeatureDimDefault = 80;       // Default feature dimension
        private const int SampleRateDefault = 16000;    // Default sample rate (Hz)
        private const int ChannelsDefault = 1;          // Default audio channels
        private const int StftWinLen = 4096;            // STFT window length
        private const int StftFftLen = 4096;            // FFT length
        private const string StftWinType = "hanning";   // STFT window type
        private const int StftWinInc = 1024;            // STFT window increment
        private const int FullFreqBins = 2049;          // Total frequency bins for complex spectrum
        private const int CropFreqBins = 1024;          // Frequency bins after cropping
        private const float TrimDuration = 0.5f;        // Audio tail trimming duration (seconds)
        #endregion

        #region Private Fields
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim;
        private int _sampleRate;
        private int _channels;
        private int _chunkLength;
        private int _shiftLength;
        #endregion

        #region Constructor
        /// <summary>
        /// Initializes a new instance of the SepProjOfSpleeter class
        /// </summary>
        /// <param name="sepModel">Separation model wrapper</param>
        /// <exception cref="ArgumentNullException">Thrown when sepModel is null</exception>
        public SepProjOfSpleeter(SepModel sepModel)
        {
            _modelSession = sepModel?.ModelSession ?? throw new ArgumentNullException(nameof(sepModel), "Separation model cannot be null");
            _customMetadata = sepModel.CustomMetadata;
            _featureDim = sepModel.FeatureDim > 0 ? sepModel.FeatureDim : FeatureDimDefault;
            _sampleRate = sepModel.SampleRate > 0 ? sepModel.SampleRate : SampleRateDefault;
            _channels = sepModel.Channels > 0 ? sepModel.Channels : ChannelsDefault;
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
        /// Converts complex spectrogram to STFT format (real + imaginary parts)
        /// </summary>
        /// <param name="complexArray">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [freq_bins, time_frames, 2]</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexArray)
        {
            int freqBins = complexArray.GetLength(0);   // Frequency bins (e.g., 961)
            int timeFrames = complexArray.GetLength(2); // Time frames (e.g., 1808)

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
        /// Converts STFT format array to complex spectrogram using conjugate symmetry
        /// </summary>
        /// <param name="stftFormat">Input STFT data with shape [1024, time_frames, 2]</param>
        /// <returns>Complex spectrogram with shape [2049, 1, time_frames]</returns>
        /// <exception cref="ArgumentException">Thrown when input has invalid dimensions</exception>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int inputFreqBins = stftFormat.GetLength(0);
            int timeFrames = stftFormat.GetLength(1);

            // Validate input dimensions
            if (inputFreqBins != CropFreqBins || stftFormat.GetLength(2) != 2)
                throw new ArgumentException($"Input array must be in [{CropFreqBins}, time_frames, 2] format");

            // Create target complex array [2049, 1, time_frames]
            Complex[,,] complexSpectrogram = new Complex[FullFreqBins, 1, timeFrames];

            // Process each time frame
            for (int t = 0; t < timeFrames; t++)
            {
                // 1. Copy positive frequency components (0-1023)
                for (int f = 0; f < CropFreqBins; f++)
                {
                    float real = stftFormat[f, t, 0];
                    float imag = stftFormat[f, t, 1];
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }

                // 2. Generate negative frequency components (1025-2048) using conjugate symmetry
                for (int f = 1025; f < FullFreqBins; f++)
                {
                    int posFreqIndex = 2048 - (f - 1024);
                    Complex posFreqValue = complexSpectrogram[posFreqIndex, 0, t];
                    complexSpectrogram[f, 0, t] = Complex.Conjugate(posFreqValue);
                }

                // 3. Special handling for Nyquist frequency (index 1024)
                // For real signals, imaginary part at Nyquist frequency should be 0
                complexSpectrogram[1024, 0, t] = new Complex(stftFormat[1023, t, 0], 0);
            }

            return complexSpectrogram;
        }
        #endregion

        #region Audio Channel Processing
        /// <summary>
        /// Splits stereo audio sample into left and right mono channels
        /// </summary>
        /// <param name="sample">Stereo audio sample array (interleaved)</param>
        /// <returns>Tuple containing left and right channels, or null if input is invalid</returns>
        public static (float[] left, float[] right)? SplitStereoToMono(float[] sample)
        {
            if (sample == null || sample.Length % 2 != 0)
            {
                Debug.WriteLine("Error: Invalid stereo sample data (null or odd length)");
                return null;
            }

            int channelLength = sample.Length / 2;
            float[] leftChannel = new float[channelLength];
            float[] rightChannel = new float[channelLength];

            for (int n = 0; n < channelLength; n++)
            {
                leftChannel[n] = sample[n * 2];
                rightChannel[n] = sample[n * 2 + 1];
            }

            return (leftChannel, rightChannel);
        }

        /// <summary>
        /// Merges left and right mono channels into interleaved stereo audio
        /// </summary>
        /// <param name="leftChannel">Left mono channel</param>
        /// <param name="rightChannel">Right mono channel</param>
        /// <returns>Interleaved stereo audio array, or null if inputs are invalid</returns>
        public static float[]? MergeMonoToStereo(float[] leftChannel, float[] rightChannel)
        {
            if (leftChannel == null || rightChannel == null)
            {
                Debug.WriteLine("Error: Left or right channel data cannot be null");
                return null;
            }

            if (leftChannel.Length != rightChannel.Length)
            {
                Debug.WriteLine($"Error: Channel length mismatch (left: {leftChannel.Length}, right: {rightChannel.Length})");
                return null;
            }

            int stereoLength = leftChannel.Length * 2;
            float[] stereoSamples = new float[stereoLength];

            for (int i = 0; i < leftChannel.Length; i++)
            {
                stereoSamples[i * 2] = leftChannel[i];       // Left channel at even indices
                stereoSamples[i * 2 + 1] = rightChannel[i];  // Right channel at odd indices
            }

            return stereoSamples;
        }
        #endregion

        #region Model Processing Methods
        /// <summary>
        /// Performs stereo audio separation using Spleeter model
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

            int batchSize = modelInputs.Count;
            float[] samples = PadHelper.PadSequence(modelInputs);
            var modelOutputEntities = new List<ModelOutputEntity>();

            var splitResult = SplitStereoToMono(samples);
            if (!splitResult.HasValue)
                return modelOutputEntities;

            try
            {
                var (leftChannel, rightChannel) = splitResult.Value;
                var stftArgs = CreateStftArgs();

                // Process STFT for both channels
                var (stftInput, stftMagInput) = ProcessDualChannelStft(leftChannel, rightChannel, stftArgs);

                // Run model inference
                var inferenceResults = RunInference(stftInput, stftMagInput);
                if (inferenceResults == null)
                    return modelOutputEntities;

                // Process inference results into output entities
                modelOutputEntities.AddRange(ProcessInferenceResults(
                    inferenceResults, stftArgs, samples, modelInputs[0].SampleRate));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Stereo separation failed: {ex.Message}\nStack trace: {ex.StackTrace}");
            }

            return modelOutputEntities;
        }

        /// <summary>
        /// Performs mono audio separation using Spleeter model
        /// </summary>
        /// <param name="modelInputs">List of model input entities</param>
        /// <param name="statesList">List of states (optional)</param>
        /// <param name="offset">Offset value (optional)</param>
        /// <returns>List of separated audio output entities</returns>
        /// <exception cref="ArgumentNullException">Thrown when modelInputs is null or empty</exception>
        public List<ModelOutputEntity> ModelProj_mono(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            if (modelInputs == null || modelInputs.Count == 0)
                throw new ArgumentNullException(nameof(modelInputs), "Model input list cannot be null or empty");

            float[] samples = PadHelper.PadSequence(modelInputs);
            var stftArgs = CreateStftArgs();

            try
            {
                // Process mono as dual identical channels
                var (stftInput, stftMagInput) = ProcessMonoAsDualChannelStft(samples, stftArgs);

                // Run model inference
                var inferenceResults = RunInference(stftInput, stftMagInput);
                if (inferenceResults == null)
                    return new List<ModelOutputEntity>();

                // Process inference results
                return ProcessMonoInferenceResults(
                    inferenceResults, stftArgs, samples, modelInputs[0].SampleRate);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Mono separation failed: {ex.Message}\nStack trace: {ex.StackTrace}");
                return new List<ModelOutputEntity>();
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
        private STFTArgs CreateStftArgs()
        {
            return new STFTArgs
            {
                WinLen = StftWinLen,
                FftLen = StftFftLen,
                WinType = StftWinType,
                WinInc = StftWinInc
            };
        }

        /// <summary>
        /// Processes stereo channels through STFT and prepares model inputs
        /// </summary>
        private (float[] stftInput, float[] magInput) ProcessDualChannelStft(
            float[] leftChannel, float[] rightChannel, STFTArgs stftArgs)
        {
            // Compute STFT for both channels
            Complex[,,] stftLeft = AudioProcessing.Stft(leftChannel, stftArgs, normalized: false, padMode: "constant");
            Complex[,,] stftRight = AudioProcessing.Stft(rightChannel, stftArgs, normalized: false, padMode: "constant");

            // Convert to STFT format and merge
            float[,,] spectrumLeft = ConvertComplexToSTFTFormat(stftLeft);
            float[,,] spectrumRight = ConvertComplexToSTFTFormat(stftRight);
            float[,,,] mergedStft = MergeSpectrums(spectrumLeft, spectrumRight);

            // Crop frequencies and compute magnitude
            float[,,,] croppedStft = CropSTFTFrequencies(mergedStft, CropFreqBins);
            float[,,] magnitude = ProcessSTFTAndComputeMagnitude(croppedStft, CropFreqBins);

            return (croppedStft.Cast<float>().ToArray(), magnitude.Cast<float>().ToArray());
        }

        /// <summary>
        /// Processes mono channel as dual identical channels for stereo model input
        /// </summary>
        private (float[] stftInput, float[] magInput) ProcessMonoAsDualChannelStft(
            float[] monoChannel, STFTArgs stftArgs)
        {
            Complex[,,] stftComplex = AudioProcessing.Stft(monoChannel, stftArgs, normalized: false, padMode: "constant");
            float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);

            // Merge with itself to simulate stereo input
            float[,,,] mergedStft = MergeSpectrums(spectrum, spectrum);
            float[,,,] croppedStft = CropSTFTFrequencies(mergedStft, CropFreqBins);
            float[,,] magnitude = ProcessSTFTAndComputeMagnitude(croppedStft, CropFreqBins);

            return (croppedStft.Cast<float>().ToArray(), magnitude.Cast<float>().ToArray());
        }

        /// <summary>
        /// Runs ONNX model inference with prepared inputs
        /// </summary>
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunInference(
            float[] stftInput, float[] magInput)
        {
            if (_modelSession == null)
            {
                Debug.WriteLine("Error: Model session is not initialized");
                return null;
            }

            var inputContainer = new List<NamedOnnxValue>();
            var inputMeta = _modelSession.InputMetadata;

            foreach (var inputName in inputMeta.Keys)
            {
                if (inputName == "input")
                {
                    int frameCount = stftInput.Length / 2 / CropFreqBins / 2;
                    var inputTensor = new DenseTensor<float>(stftInput,
                        new[] { 2, CropFreqBins, frameCount, 2 }, false);
                    inputContainer.Add(NamedOnnxValue.CreateFromTensor(inputName, inputTensor));
                }
                else if (inputName == "input_mag")
                {
                    int frameCount = magInput.Length / 2 / CropFreqBins;
                    var magTensor = new DenseTensor<float>(magInput,
                        new[] { 2, CropFreqBins, frameCount }, false);
                    inputContainer.Add(NamedOnnxValue.CreateFromTensor(inputName, magTensor));
                }
            }

            return _modelSession.Run(inputContainer);
        }

        /// <summary>
        /// Processes stereo inference results into output entities
        /// </summary>
        private IEnumerable<ModelOutputEntity> ProcessInferenceResults(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            STFTArgs stftArgs, float[] originalSamples, int sampleRate)
        {
            foreach (var result in results)
            {
                var outputTensor = result.AsTensor<float>();
                var (channel0, channel1) = SplitStereoSTFT(outputTensor);

                // Process left channel
                float[,,] spec0 = To3DArray(channel0);
                Complex[,,] spectrum0 = ConvertSTFTFormatToComplex(spec0);
                float[] output0 = AudioProcessing.Istft(spectrum0, stftArgs, originalSamples.Length, normalized: false);
                float[] trimmedLeft = TrimAudio(output0, sampleRate, 0.5f);

                // Process right channel
                float[,,] spec1 = To3DArray(channel1);
                Complex[,,] spectrum1 = ConvertSTFTFormatToComplex(spec1);
                float[] output1 = AudioProcessing.Istft(spectrum1, stftArgs, originalSamples.Length, normalized: false);
                float[] trimmedRight = TrimAudio(output1, sampleRate, 0.5f);

                // Merge to stereo
                float[] stereoOutput = MergeMonoToStereo(trimmedLeft, trimmedRight);
                if (stereoOutput != null)
                {
                    yield return new ModelOutputEntity
                    {
                        StemName = result.Name,
                        StemContents = stereoOutput
                    };
                }
            }
        }

        /// <summary>
        /// Processes mono inference results into output entities
        /// </summary>
        private List<ModelOutputEntity> ProcessMonoInferenceResults(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            STFTArgs stftArgs, float[] originalSamples, int sampleRate)
        {
            var outputs = new List<ModelOutputEntity>();

            foreach (var result in results)
            {
                var outputTensor = result.AsTensor<float>();
                var (channel0, _) = SplitStereoSTFT(outputTensor);

                float[,,] spec = To3DArray(channel0);
                Complex[,,] spectrum = ConvertSTFTFormatToComplex(spec);
                float[] output = AudioProcessing.Istft(spectrum, stftArgs, originalSamples.Length, normalized: false);

                outputs.Add(new ModelOutputEntity
                {
                    StemName = result.Name,
                    StemContents = output
                });
            }

            return outputs;
        }

        /// <summary>
        /// Trims excess samples from audio tail
        /// </summary>
        private float[] TrimAudio(float[] audio, int sampleRate, float trimSeconds)
        {
            int trimSampleCount = (int)(sampleRate * trimSeconds);
            int outputLength = Math.Max(0, audio.Length - trimSampleCount);

            float[] trimmed = new float[outputLength];
            Array.Copy(audio, 0, trimmed, 0, outputLength);

            return trimmed;
        }
        #endregion

        #region Tensor/Array Processing Utilities
        /// <summary>
        /// Converts a 3D tensor to a 3D float array
        /// </summary>
        /// <param name="tensor">Input 3D tensor</param>
        /// <returns>3D float array with the same dimensions</returns>
        /// <exception cref="ArgumentException">Thrown when tensor is not 3-dimensional</exception>
        public float[,,] To3DArray(Tensor<float> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional");

            var dimensions = tensor.Dimensions;
            float[,,] result = new float[dimensions[0], dimensions[1], dimensions[2]];
            int[] indices = new int[3];

            for (indices[0] = 0; indices[0] < dimensions[0]; indices[0]++)
            {
                for (indices[1] = 0; indices[1] < dimensions[1]; indices[1]++)
                {
                    for (indices[2] = 0; indices[2] < dimensions[2]; indices[2]++)
                    {
                        result[indices[0], indices[1], indices[2]] = tensor[indices];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Converts a 3D float array to a 2D complex array (with zero imaginary parts)
        /// </summary>
        /// <param name="floatArray">Input 3D array with shape [1, rows, cols]</param>
        /// <returns>2D complex array with shape [rows, cols]</returns>
        /// <exception cref="ArgumentException">Thrown when input has invalid dimensions</exception>
        public static Complex[,] ConvertToComplex(float[,,] floatArray)
        {
            if (floatArray.Rank != 3 || floatArray.GetLength(0) != 1)
                throw new ArgumentException("Input array must be 3-dimensional with first dimension length 1");

            int rows = floatArray.GetLength(1);
            int cols = floatArray.GetLength(2);
            Complex[,] complexArray = new Complex[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    complexArray[i, j] = new Complex(floatArray[0, i, j], 0);
                }
            }

            return complexArray;
        }

        /// <summary>
        /// Merges two 3D spectrums into a 4D array (adding channel dimension)
        /// </summary>
        /// <param name="spectrum1">First spectrum with shape [freq_bins, time_frames, 2]</param>
        /// <param name="spectrum2">Second spectrum with shape [freq_bins, time_frames, 2]</param>
        /// <returns>Merged 4D array with shape [2, freq_bins, time_frames, 2]</returns>
        /// <exception cref="ArgumentException">Thrown when input arrays are not 3-dimensional</exception>
        public static float[,,,] MergeSpectrums(float[,,] spectrum1, float[,,] spectrum2)
        {
            if (spectrum1.Rank != 3 || spectrum2.Rank != 3)
                throw new ArgumentException("Input arrays must be 3-dimensional");

            int freqBins = spectrum1.GetLength(0);
            int timeFrames = spectrum1.GetLength(1);
            int complexParts = spectrum1.GetLength(2);

            float[,,,] merged = new float[2, freqBins, timeFrames, complexParts];

            // Copy first spectrum to channel 0
            for (int f = 0; f < freqBins; f++)
                for (int t = 0; t < timeFrames; t++)
                    for (int c = 0; c < complexParts; c++)
                        merged[0, f, t, c] = spectrum1[f, t, c];

            // Copy second spectrum to channel 1
            for (int f = 0; f < freqBins; f++)
                for (int t = 0; t < timeFrames; t++)
                    for (int c = 0; c < complexParts; c++)
                        merged[1, f, t, c] = spectrum2[f, t, c];

            return merged;
        }

        /// <summary>
        /// Crops STFT frequency range (equivalent to Python: stft = stft[:, :maxFreq, :, :])
        /// </summary>
        /// <param name="stft">Input 4D STFT array [channels, freq, time, complex_parts]</param>
        /// <param name="maxFreq">Maximum frequency index to retain (inclusive)</param>
        /// <returns>Cropped 4D STFT array [channels, maxFreq, time, complex_parts]</returns>
        /// <exception cref="ArgumentException">Thrown when maxFreq exceeds original frequency range</exception>
        public static float[,,,] CropSTFTFrequencies(float[,,,] stft, int maxFreq)
        {
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);
            int complexParts = stft.GetLength(3);

            if (maxFreq >= originalFreqBins)
                throw new ArgumentException($"maxFreq ({maxFreq}) must be less than original frequency bins ({originalFreqBins})");

            float[,,,] cropped = new float[numChannels, maxFreq, timeFrames, complexParts];

            for (int ch = 0; ch < numChannels; ch++)
                for (int f = 0; f < maxFreq; f++)
                    for (int t = 0; t < timeFrames; t++)
                        for (int c = 0; c < complexParts; c++)
                            cropped[ch, f, t, c] = stft[ch, f, t, c];

            return cropped;
        }

        /// <summary>
        /// Processes STFT and computes magnitude spectrum (sqrt(real² + imag²))
        /// </summary>
        /// <param name="stft">Input 4D STFT array [channels, freq, time, complex_parts]</param>
        /// <param name="maxFreq">Maximum frequency index to process</param>
        /// <returns>3D magnitude array [channels, maxFreq, time]</returns>
        /// <exception cref="ArgumentException">Thrown when maxFreq exceeds original frequency range</exception>
        public static float[,,] ProcessSTFTAndComputeMagnitude(float[,,,] stft, int maxFreq)
        {
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);

            if (maxFreq > originalFreqBins)
                throw new ArgumentException($"maxFreq ({maxFreq}) exceeds original frequency bins ({originalFreqBins})");

            float[,,] magnitude = new float[numChannels, maxFreq, timeFrames];

            for (int ch = 0; ch < numChannels; ch++)
                for (int f = 0; f < maxFreq; f++)
                    for (int t = 0; t < timeFrames; t++)
                    {
                        float real = stft[ch, f, t, 0];
                        float imag = stft[ch, f, t, 1];
                        magnitude[ch, f, t] = (float)Math.Sqrt(real * real + imag * imag);
                    }

            return magnitude;
        }

        /// <summary>
        /// Splits 4D STFT array [2,1024,time,2] into two 3D arrays [1024,time,2]
        /// </summary>
        /// <param name="stft">Input 4D STFT array</param>
        /// <returns>Tuple containing two 3D arrays for each channel</returns>
        /// <exception cref="ArgumentException">Thrown when input has invalid dimensions</exception>
        public static (float[,,] channel0, float[,,] channel1) SplitStereoSTFT(float[,,,] stft)
        {
            if (stft.Rank != 4 || stft.GetLength(0) != 2 ||
                stft.GetLength(1) != CropFreqBins || stft.GetLength(3) != 2)
                throw new ArgumentException($"Input array must be in [2, {CropFreqBins}, time, 2] format");

            int timeFrames = stft.GetLength(2);
            float[,,] channel0 = new float[CropFreqBins, timeFrames, 2];
            float[,,] channel1 = new float[CropFreqBins, timeFrames, 2];

            for (int f = 0; f < CropFreqBins; f++)
                for (int t = 0; t < timeFrames; t++)
                    for (int c = 0; c < 2; c++)
                    {
                        channel0[f, t, c] = stft[0, f, t, c];
                        channel1[f, t, c] = stft[1, f, t, c];
                    }

            return (channel0, channel1);
        }

        /// <summary>
        /// Splits 4D STFT tensor [2,1024,time,2] into two 3D tensors [1024,time,2]
        /// </summary>
        /// <param name="stftTensor">Input 4D STFT tensor</param>
        /// <returns>Tuple containing two 3D tensors for each channel</returns>
        /// <exception cref="ArgumentException">Thrown when input has invalid dimensions</exception>
        public static (Tensor<float> channel0, Tensor<float> channel1) SplitStereoSTFT(Tensor<float> stftTensor)
        {
            if (stftTensor.Dimensions.Length != 4 || stftTensor.Dimensions[0] != 2 ||
                stftTensor.Dimensions[1] != CropFreqBins || stftTensor.Dimensions[3] != 2)
                throw new ArgumentException($"Input tensor must be in [2, {CropFreqBins}, time, 2] format");

            int timeFrames = stftTensor.Dimensions[2];
            var channel0 = new DenseTensor<float>(new[] { CropFreqBins, timeFrames, 2 });
            var channel1 = new DenseTensor<float>(new[] { CropFreqBins, timeFrames, 2 });

            for (int f = 0; f < CropFreqBins; f++)
                for (int t = 0; t < timeFrames; t++)
                    for (int c = 0; c < 2; c++)
                    {
                        channel0[f, t, c] = stftTensor[0, f, t, c];
                        channel1[f, t, c] = stftTensor[1, f, t, c];
                    }

            return (channel0, channel1);
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
        ~SepProjOfSpleeter()
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