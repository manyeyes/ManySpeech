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
    internal class SepProjOfUvr : ISepProj, IDisposable
    {
        #region Private Fields
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        #endregion

        #region Constants
        private const int F = 2048;
        private const int T = 512;
        #endregion

        #region Constructor
        public SepProjOfUvr(SepModel sepModel)
        {
            _modelSession = sepModel.ModelSession;
            _customMetadata = sepModel.CustomMetadata;
            _featureDim = sepModel.FeatureDim;
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
            set => _modelSession = value;
        }

        public CustomMetadata CustomMetadata
        {
            get => _customMetadata;
            set => _customMetadata = value;
        }

        public int ChunkLength
        {
            get => _chunkLength;
            set => _chunkLength = value;
        }

        public int ShiftLength
        {
            get => _shiftLength;
            set => _shiftLength = value;
        }

        public int FeatureDim
        {
            get => _featureDim;
            set => _featureDim = value;
        }

        public int SampleRate
        {
            get => _sampleRate;
            set => _sampleRate = value;
        }

        public int Channels
        {
            get => _channels;
            set => _channels = value;
        }
        #endregion

        #region State Handling Methods
        /// <summary>
        /// Stacks a list of state lists into a single state list
        /// </summary>
        /// <param name="statesList">List of state lists to stack</param>
        /// <returns>Stacked state list</returns>
        public List<float[]> StackStates(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            states = statesList[0];
            return states;
        }

        /// <summary>
        /// Unstacks a single state list into a list of state lists
        /// </summary>
        /// <param name="states">State list to unstack</param>
        /// <returns>Unstacked list of state lists</returns>
        public List<List<float[]>> UnstackStates(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 2 == 0, "When stacking states, state_list[0] should have even count");
            statesList.Add(states);
            return statesList;
        }
        #endregion

        #region Spectrogram Conversion Methods
        /// <summary>
        /// Converts a Complex[961, 1, 1808] array to float[961, 1808, 2] STFT format
        /// </summary>
        /// <param name="complexArray">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [freq_bins, time_frames, 2]</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexArray)
        {
            int freqBins = complexArray.GetLength(0);   // 961 (frequency bins)
            int channels = complexArray.GetLength(1);   // 1 (single channel)
            int timeFrames = complexArray.GetLength(2); // 1808 (time frames)

            // Target shape: [freqBins, timeFrames, 2] (real part + imaginary part)
            float[,,] floatArray = new float[freqBins, timeFrames, 2];

            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // Extract real and imaginary parts
                    floatArray[f, t, 0] = (float)complexArray[f, 0, t].Real;      // Real part
                    floatArray[f, t, 1] = (float)complexArray[f, 0, t].Imaginary; // Imaginary part
                }
            }
            return floatArray;
        }

        /// <summary>
        /// Converts float[F, T, 2] STFT format to Complex[F, 1, T] complex spectrogram
        /// </summary>
        /// <param name="stftFormat">Input STFT data with shape [frequency, time, real/imaginary]</param>
        /// <returns>Complex spectrogram with shape [frequency, 1, time]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // Number of frequency bins (F)
            int timeFrames = stftFormat.GetLength(1);   // Number of time frames (T)

            // Validate input dimensions
            if (freqBins != F || stftFormat.GetLength(2) != 2)
            {
                throw new ArgumentException("Input array must be in [F, time_frames, 2] format");
            }

            // Create target complex array [F, 1, T]
            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            // Iterate through each frequency bin and time frame
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // Get real and imaginary parts
                    float real = stftFormat[f, t, 0];
                    float imag = stftFormat[f, t, 1];

                    // Create complex number and store in output array
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }
            }

            return complexSpectrogram;
        }

        /// <summary>
        /// Converts float[F, T, 2] STFT format to Complex[F, 2, T] complex spectrogram
        /// </summary>
        /// <param name="stftFormat">Input STFT data with shape [frequency, time, real/imaginary]</param>
        /// <returns>Complex spectrogram with shape [frequency, 2 channels, time]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex2(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // Number of frequency bins (F)
            int timeFrames = stftFormat.GetLength(1);   // Number of time frames (T)

            // Create target complex array [F, 2, T]
            Complex[,,] complexSpectrogram = new Complex[freqBins, 2, timeFrames];

            // Iterate through each frequency bin and time frame
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // Get real and imaginary parts
                    float real = stftFormat[f, t, 0];
                    float imag = stftFormat[f, t, 1];

                    // Store in complex array, second dimension corresponds to channels
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                    complexSpectrogram[f, 1, t] = new Complex(real, imag); // Copy to second channel
                }
            }

            return complexSpectrogram;
        }

        /// <summary>
        /// Converts float32[batch_size,4,F,T] data to complex spectrogram representation, 
        /// returns List<float[,,]> by batch where each float[,,] has shape [F, T, 2] 
        /// corresponding to [frequency, time, real/imaginary]
        /// </summary>
        /// <param name="input">Input data with shape [batch_size, 4, F, T]</param>
        /// <returns>List of complex spectrograms, each with shape [F, T, 2]</returns>
        public static List<float[,,]> ConvertBatchSpectrums(float[,,,] input)
        {
            int batchSize = input.GetLength(0);
            int freqBins = input.GetLength(2);    // F
            int timeFrames = input.GetLength(3);  // T

            // Validate input dimensions
            if (input.GetLength(1) != 4)
            {
                throw new ArgumentException("The second dimension of input array must be 4, representing real/imaginary parts of two channels");
            }

            // Initialize result list
            List<float[,,]> result = new List<float[,,]>(batchSize);

            // Process each batch sample
            for (int b = 0; b < batchSize; b++)
            {
                // Create new 3D array [F, T, 2]
                float[,,] complexSpectrum = new float[freqBins, timeFrames, 2];

                // Iterate through each frequency and time point
                for (int f = 0; f < freqBins; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // Extract data from 4 channels
                        // Assuming channel order: left real, left imaginary, right real, right imaginary
                        float leftReal = input[b, 0, f, t];   // Left channel real part
                        float leftImag = input[b, 1, f, t];   // Left channel imaginary part
                        float rightReal = input[b, 2, f, t];  // Right channel real part
                        float rightImag = input[b, 3, f, t];  // Right channel imaginary part

                        // Merge left and right channels (take average)
                        float realPart = (leftReal + rightReal) / 2.0f;
                        float imagPart = (leftImag + rightImag) / 2.0f;

                        // Store in result array [frequency, time, real/imaginary]
                        complexSpectrum[f, t, 0] = realPart;   // Real part
                        complexSpectrum[f, t, 1] = imagPart;   // Imaginary part
                    }
                }

                // Add to result list
                result.Add(complexSpectrum);
            }

            return result;
        }
        #endregion

        #region Audio Channel Processing
        /// <summary>
        /// Splits stereo audio sample into mono left and right channels
        /// </summary>
        /// <param name="sample">Stereo audio sample array</param>
        /// <returns>Tuple containing left and right channels, or null if input is invalid</returns>
        public static (float[] left, float[] right)? SplitStereoToMono(float[] sample)
        {
            if (sample == null || sample.Length % 2 != 0)
            {
                Console.WriteLine("Error: Invalid stereo sample data");
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
        /// Merges mono left and right channels into stereo audio
        /// </summary>
        /// <param name="leftChannel">Left mono channel</param>
        /// <param name="rightChannel">Right mono channel</param>
        /// <returns>Merged stereo audio array, or null if input is invalid</returns>
        public static float[]? MergeMonoToStereo(float[] leftChannel, float[] rightChannel)
        {
            // Validate input
            if (leftChannel == null || rightChannel == null)
            {
                Console.WriteLine("Error: Left or right channel data cannot be null");
                return null;
            }

            if (leftChannel.Length != rightChannel.Length)
            {
                Console.WriteLine($"Error: Mismatched channel lengths (left: {leftChannel.Length}, right: {rightChannel.Length})");
                return null;
            }

            int stereoLength = leftChannel.Length * 2;
            float[] stereoSamples = new float[stereoLength];

            // Merge left and right channel data
            for (int i = 0; i < leftChannel.Length; i++)
            {
                stereoSamples[i * 2] = leftChannel[i];         // Left channel at even indices
                stereoSamples[i * 2 + 1] = rightChannel[i];    // Right channel at odd indices
            }

            return stereoSamples;
        }
        #endregion

        #region Model Projection Methods
        /// <summary>
        /// Performs model projection for audio separation
        /// </summary>
        /// <param name="modelInputs">List of model input entities</param>
        /// <param name="statesList">List of states (optional)</param>
        /// <param name="offset">Offset value (optional)</param>
        /// <returns>List of model output entities</returns>
        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            int chunkSize = ((T - 1) * 1024 + F * 2 - 1) * 2;
            int tailLen = chunkSize - modelInputs[0].Speech.Length;
            long[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs, tailLen: tailLen);
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();

            var splitResult = SplitStereoToMono(samples);
            if (splitResult.HasValue)
            {
                float[] leftChannel = splitResult.Value.left;
                float[] rightChannel = splitResult.Value.right;

                STFTArgs args = new STFTArgs();
                args.WinLen = F * 2 - 1;
                args.FftLen = F * 2 - 1;
                args.WinType = "hanning";
                args.WinInc = 1024;

                try
                {
                    // Perform STFT on audio
                    Complex[,,] stftComplexLeft = AudioProcessing.Stft(leftChannel, args, normalized: false);
                    float[,,] spectrumLeft = ConvertComplexToSTFTFormat(stftComplexLeft);

                    Complex[,,] stftComplexRight = AudioProcessing.Stft(rightChannel, args, normalized: false);
                    float[,,] spectrumRight = ConvertComplexToSTFTFormat(stftComplexRight);

                    float[,,] stft = MergeSpectrums(spectrumLeft, spectrumRight);
                    float[] input = stft.Cast<float>().ToArray();

                    var inputMeta = _modelSession.InputMetadata;
                    var container = new List<NamedOnnxValue>();

                    foreach (var name in inputMeta.Keys)
                    {
                        if (name == "input")
                        {
                            int[] dim = new int[] { batchSize, 4, F, T };
                            var tensor = new DenseTensor<float>(input, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }
                    }

                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = _modelSession.Run(container);

                    if (encoderResults != null)
                    {
                        foreach (var encoderResult in encoderResults)
                        {
                            string name = encoderResult.Name;
                            var outputTensor = encoderResult.AsTensor<float>();

                            var tensorList = ConvertTensorToList(outputTensor);
                            int sampleRate = modelInputs[0].SampleRate;

                            (Tensor<float> channel0, Tensor<float> channel1) channels = SplitStereoSTFT(tensorList[0]);

                            var spec0 = To3DArray(channels.channel0);
                            Complex[,,] spectrumX0 = ConvertSTFTFormatToComplex(spec0);
                            float[] output0 = AudioProcessing.Istft(spectrumX0, args, samples.Length, normalized: false);

                            float[] left = new float[(int)(samples.Length - tailLen - sampleRate * 0.5f) / 2];
                            Array.Copy(output0, 0, left, 0, left.Length);

                            var spec1 = To3DArray(channels.channel1);
                            Complex[,,] spectrumX1 = ConvertSTFTFormatToComplex(spec1);
                            float[] output1 = AudioProcessing.Istft(spectrumX1, args, samples.Length, normalized: false);

                            float[] right = new float[(int)(samples.Length - tailLen - sampleRate * 0.5f) / 2];
                            Array.Copy(output1, 0, right, 0, right.Length);

                            float[] output = MergeMonoToStereo(left, right);

                            modelOutputEntities.Add(new ModelOutputEntity
                            {
                                StemName = name,
                                StemContents = output
                            });
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error in ModelProj: {ex.Message}");
                }
            }

            return modelOutputEntities;
        }

        /// <summary>
        /// Performs mono model projection for audio separation
        /// </summary>
        /// <param name="modelInputs">List of model input entities</param>
        /// <param name="statesList">List of states (optional)</param>
        /// <param name="offset">Offset value (optional)</param>
        /// <returns>List of model output entities</returns>
        public List<ModelOutputEntity> ModelProj_mono(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            long[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);

            STFTArgs args = new STFTArgs();
            args.WinLen = 4096;
            args.FftLen = 4096;
            args.WinType = "hanning";
            args.WinInc = 1024;

            // Perform STFT on audio
            Complex[,,] stftComplex = AudioProcessing.Stft(samples, args, normalized: false, padMode: "constant");
            float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
            float[,,] stft = MergeSpectrums(spectrum, spectrum);

            float[] input = stft.Cast<float>().ToArray();
            var inputMeta = _modelSession.InputMetadata;
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            var container = new List<NamedOnnxValue>();

            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { 1, 4, F, T };
                    var tensor = new DenseTensor<float>(input, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = _modelSession.Run(container);

                if (encoderResults != null)
                {
                    foreach (var encoderResult in encoderResults)
                    {
                        string name = encoderResult.Name;
                        var outputTensor = encoderResult.AsTensor<float>();

                        (Tensor<float> channel0, Tensor<float> channel1) channels = SplitStereoSTFT(outputTensor);
                        var spec = To3DArray(channels.channel0);

                        Complex[,,] spectrumX = ConvertSTFTFormatToComplex(spec);
                        float[] output = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false);

                        modelOutputEntities.Add(new ModelOutputEntity
                        {
                            StemName = name,
                            StemContents = output
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error in ModelProj_mono: {ex.Message}");
            }

            return modelOutputEntities;
        }

        /// <summary>
        /// Generator projection (not implemented)
        /// </summary>
        /// <param name="modelOutputEntity">Model output entity</param>
        /// <param name="batchSize">Batch size</param>
        /// <returns>Null</returns>
        public List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1)
        {
            return null;
        }
        #endregion

        #region Tensor/Array Utilities
        /// <summary>
        /// Converts a 3D Tensor to a 3D float array
        /// </summary>
        /// <param name="tensor">Input 3D tensor</param>
        /// <returns>3D float array with the same dimensions</returns>
        /// <exception cref="ArgumentException">Thrown if tensor is not 3-dimensional</exception>
        public float[,,] To3DArray(Tensor<float> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional");

            var dimensions = tensor.Dimensions;
            float[,,] result = new float[dimensions[0], dimensions[1], dimensions[2]];

            // Universal index access
            var indices = new int[3];
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
        /// Converts a float32[batch_size,4,F,T] Tensor to List<Tensor<float>> 
        /// where each Tensor has shape [F, T, 4] corresponding to [frequency, time, channel]
        /// </summary>
        /// <param name="tensor">Input Tensor with shape [batch_size, 4, F, T]</param>
        /// <returns>List of complex spectrograms, each as a [F, T, 4] Tensor</returns>
        /// <exception cref="ArgumentException">Thrown if tensor is not 4-dimensional</exception>
        public List<Tensor<float>> ConvertTensorToList(Tensor<float> tensor)
        {
            if (tensor.Rank != 4)
                throw new ArgumentException("Tensor must be 4-dimensional with shape [batch_size, 4, F, T]");

            int batchSize = tensor.Dimensions[0];
            int channels = tensor.Dimensions[1];     // 4
            int freqBins = tensor.Dimensions[2];     // F
            int timeFrames = tensor.Dimensions[3];   // T

            // Initialize result list
            List<Tensor<float>> result = new List<Tensor<float>>(batchSize);

            // Process each batch sample
            for (int b = 0; b < batchSize; b++)
            {
                // Create new 3D Tensor [F, T, 4]
                var sampleTensor = new DenseTensor<float>(new int[] { freqBins, timeFrames, channels });

                // Universal index access
                var indices = new int[4];
                indices[0] = b; // Set batch index

                for (int f = 0; f < freqBins; f++)
                {
                    indices[2] = f; // Set frequency index

                    for (int t = 0; t < timeFrames; t++)
                    {
                        indices[3] = t; // Set time index

                        for (int c = 0; c < channels; c++)
                        {
                            indices[1] = c; // Set channel index

                            // Get value from input Tensor and assign to new Tensor
                            sampleTensor[f, t, c] = tensor[indices];
                        }
                    }
                }

                // Add to result list
                result.Add(sampleTensor);
            }

            return result;
        }

        /// <summary>
        /// Converts a float[,,] array to a Complex[,] array
        /// </summary>
        /// <param name="floatArray">Input 3D float array</param>
        /// <returns>2D complex array</returns>
        /// <exception cref="ArgumentException">Thrown if input has invalid dimensions</exception>
        public static Complex[,] ConvertToComplex(float[,,] floatArray)
        {
            // Check input array dimensions
            if (floatArray.Rank != 3 || floatArray.GetLength(0) != 1)
            {
                throw new ArgumentException("Input array must be 3-dimensional with first dimension length 1.");
            }

            int rows = floatArray.GetLength(1);
            int cols = floatArray.GetLength(2);

            // Create target complex array
            Complex[,] complexArray = new Complex[rows, cols];

            // Iterate and convert each element
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // Get real part from input array, set imaginary part to 0
                    float real = floatArray[0, i, j];
                    complexArray[i, j] = new Complex(real, 0);
                }
            }

            return complexArray;
        }

        /// <summary>
        /// Merges two float[,,] spectrograms into one float[,,] (interleaving real and imaginary parts)
        /// </summary>
        /// <param name="spectrum1">First spectrogram with shape [F, T, 2]</param>
        /// <param name="spectrum2">Second spectrogram with shape [F, T, 2]</param>
        /// <returns>Merged 3D array with shape [4, F, T]</returns>
        public static float[,,] MergeSpectrums(float[,,] spectrum1, float[,,] spectrum2)
        {
            // Validate input array dimensions
            if (spectrum1.Rank != 3 || spectrum2.Rank != 3)
                throw new ArgumentException("Input arrays must be 3-dimensional");

            // Get array dimensions (assuming both arrays have the same dimensions)
            int freqBins = spectrum1.GetLength(0);    // F
            int timeFrames = spectrum1.GetLength(1);  // T
            int complexParts = spectrum1.GetLength(2); // 2

            // Validate input array dimensions match
            if (spectrum2.GetLength(0) != freqBins ||
                spectrum2.GetLength(1) != timeFrames ||
                spectrum2.GetLength(2) != complexParts)
            {
                throw new ArgumentException("The two input arrays must have matching dimensions");
            }

            // Create new 3D array [4, F, T]
            float[,,] mergedSpectrum = new float[4, freqBins, timeFrames];

            // Copy real parts of first spectrum to channel 0
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[0, f, t] = spectrum1[f, t, 0]; // Real part
                }
            }

            // Copy imaginary parts of first spectrum to channel 1
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[1, f, t] = spectrum1[f, t, 1]; // Imaginary part
                }
            }

            // Copy real parts of second spectrum to channel 2
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[2, f, t] = spectrum2[f, t, 0]; // Real part
                }
            }

            // Copy imaginary parts of second spectrum to channel 3
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[3, f, t] = spectrum2[f, t, 1]; // Imaginary part
                }
            }

            return mergedSpectrum;
        }

        /// <summary>
        /// Crops the frequency range of STFT spectrogram
        /// Equivalent to Python: stft = stft[:, :self.F, :, :]
        /// </summary>
        /// <param name="stft">Input STFT spectrogram, 4D array [channels, frequency, time, real/imaginary]</param>
        /// <param name="maxFreq">Maximum frequency index to keep (inclusive)</param>
        /// <returns>Cropped STFT spectrogram, 4D array [channels, maxFreq, time, real/imaginary]</returns>
        public static float[,,,] CropSTFTFrequencies(float[,,,] stft, int maxFreq)
        {
            // Get input dimensions
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);
            int complexParts = stft.GetLength(3); // Usually 2 (real and imaginary parts)

            // Ensure maxFreq does not exceed original frequency range
            if (maxFreq >= originalFreqBins)
                throw new ArgumentException($"maxFreq({maxFreq}) must be less than original frequency range({originalFreqBins})");

            // Create cropped array [channels, maxFreq, time, real/imaginary]
            float[,,,] croppedStft = new float[numChannels, maxFreq, timeFrames, complexParts];

            // Copy data (keep only first maxFreq frequency bins)
            for (int ch = 0; ch < numChannels; ch++)
            {
                for (int f = 0; f < maxFreq; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        for (int c = 0; c < complexParts; c++)
                        {
                            croppedStft[ch, f, t, c] = stft[ch, f, t, c];
                        }
                    }
                }
            }

            return croppedStft;
        }

        /// <summary>
        /// Processes STFT spectrogram and computes magnitude spectrum
        /// Equivalent to Python:
        /// stft = stft[:, :self.F, :, :]
        /// real = stft[:, :, :, 0]
        /// im = stft[:, :, :, 1]
        /// mag = torch.sqrt(real ** 2 + im ** 2)
        /// </summary>
        /// <param name="stft">Input STFT spectrogram, 4D array [channels, frequency, time, real/imaginary]</param>
        /// <param name="maxFreq">Maximum frequency index to keep (corresponds to self.F in Python)</param>
        /// <returns>Magnitude spectrum, 3D array [channels, frequency, time]</returns>
        public static float[,,] ProcessSTFTAndComputeMagnitude(float[,,,] stft, int maxFreq)
        {
            // Get input dimensions
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);

            // Ensure maxFreq does not exceed original frequency range
            if (maxFreq > originalFreqBins)
                throw new ArgumentException($"maxFreq({maxFreq}) exceeds original frequency range({originalFreqBins})");

            // Create output magnitude spectrum array [channels, maxFreq, time]
            float[,,] magnitude = new float[numChannels, maxFreq, timeFrames];

            // Process STFT and compute magnitude spectrum
            for (int ch = 0; ch < numChannels; ch++)
            {
                for (int f = 0; f < maxFreq; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // Get real and imaginary parts
                        float real = stft[ch, f, t, 0];
                        float imag = stft[ch, f, t, 1];

                        // Compute magnitude: sqrt(real² + imag²)
                        magnitude[ch, f, t] = (float)Math.Sqrt(real * real + imag * imag);
                    }
                }
            }

            return magnitude;
        }

        /// <summary>
        /// Splits a 3D float array [F,T,4] into two 3D float arrays [F,T,2]
        /// Corresponding to left and right channels' complex spectrograms (real and imaginary parts)
        /// </summary>
        /// <param name="inputArray">Input 3D array [F,T,4]</param>
        /// <returns>Tuple containing two 3D arrays, representing left and right channels</returns>
        public static (float[,,] channel0, float[,,] channel1) SplitStereoSTFT(float[,,] inputArray)
        {
            // Validate input dimensions
            if (inputArray.Rank != 3 ||
                inputArray.GetLength(0) != F ||
                inputArray.GetLength(1) != T ||
                inputArray.GetLength(2) != 4)
            {
                throw new ArgumentException("Input array must be a 3D array in [F,T,4] format");
            }

            // Create two 3D arrays, each representing one channel's complex spectrogram
            float[,,] leftChannel = new float[F, T, 2];
            float[,,] rightChannel = new float[F, T, 2];

            // Copy data
            for (int f = 0; f < F; f++)
            {
                for (int t = 0; t < T; t++)
                {
                    // Copy from input array's channels 0 and 1 to left channel's real and imaginary parts
                    leftChannel[f, t, 0] = inputArray[f, t, 0];  // Left channel real part
                    leftChannel[f, t, 1] = inputArray[f, t, 1];  // Left channel imaginary part

                    // Copy from input array's channels 2 and 3 to right channel's real and imaginary parts
                    rightChannel[f, t, 0] = inputArray[f, t, 2];  // Right channel real part
                    rightChannel[f, t, 1] = inputArray[f, t, 3];  // Right channel imaginary part
                }
            }

            return (leftChannel, rightChannel);
        }

        /// <summary>
        /// Splits a 3D Tensor<float>[F,T,4] into two 3D Tensor<float>[F,T,2]
        /// Corresponding to left and right channels' complex spectrograms (real and imaginary parts)
        /// </summary>
        /// <param name="inputTensor">Input 3D Tensor [F,T,4]</param>
        /// <returns>Tuple containing two 3D Tensors, representing left and right channels</returns>
        public static (Tensor<float> leftChannel, Tensor<float> rightChannel) SplitStereoSTFT(Tensor<float> inputTensor)
        {
            // Validate input dimensions
            if (inputTensor.Dimensions.Length != 3 ||
                inputTensor.Dimensions[0] != F ||
                inputTensor.Dimensions[1] != T ||
                inputTensor.Dimensions[2] != 4)
            {
                throw new ArgumentException("Input Tensor must be a 3D tensor in [F,T,4] format");
            }

            // Create two 3D Tensors, each representing one channel's complex spectrogram
            var leftChannel = new DenseTensor<float>(new int[] { F, T, 2 });
            var rightChannel = new DenseTensor<float>(new int[] { F, T, 2 });

            // Copy data
            for (int f = 0; f < F; f++)
            {
                for (int t = 0; t < T; t++)
                {
                    // Copy from input Tensor's channels 0 and 1 to left channel's real and imaginary parts
                    leftChannel[f, t, 0] = inputTensor[f, t, 0];  // Left channel real part
                    leftChannel[f, t, 1] = inputTensor[f, t, 1];  // Left channel imaginary part

                    // Copy from input Tensor's channels 2 and 3 to right channel's real and imaginary parts
                    rightChannel[f, t, 0] = inputTensor[f, t, 2];  // Right channel real part
                    rightChannel[f, t, 1] = inputTensor[f, t, 3];  // Right channel imaginary part
                }
            }

            return (leftChannel, rightChannel);
        }
        #endregion

        #region IDisposable Implementation
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    _modelSession?.Dispose();
                    _modelSession = null;
                }

                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        ~SepProjOfUvr()
        {
            Dispose(disposing: false);
        }
        #endregion
    }
}