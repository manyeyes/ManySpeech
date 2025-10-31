using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Numerics;

namespace ManySpeech.AudioSep.Utils
{
    /// <summary>
    /// Configuration parameters for Short-Time Fourier Transform (STFT)
    /// </summary>
    public class StftParameters
    {
        /// <summary>
        /// Type of window function (e.g., "hamming", "hanning")
        /// </summary>
        public string WindowType { get; set; }

        /// <summary>
        /// Length of the window in samples
        /// </summary>
        public int WindowLength { get; set; }

        /// <summary>
        /// Increment (hop size) between consecutive windows in samples
        /// </summary>
        public int WindowIncrement { get; set; }

        /// <summary>
        /// Size of the FFT used in STFT
        /// </summary>
        public int FftLength { get; set; }
    }

    /// <summary>
    /// Configuration parameters for Mel spectrogram generation
    /// </summary>
    public class MelParameters
    {
        /// <summary>
        /// Size of the FFT (NFFT)
        /// </summary>
        public int FftSize { get; set; } = 1024;

        /// <summary>
        /// Number of Mel bands filters
        /// </summary>
        public int MelBandCount { get; set; } = 80;

        /// <summary>
        /// Hop size between consecutive frames in samples
        /// </summary>
        public int HopSize { get; set; } = 256;

        /// <summary>
        /// Window size in samples
        /// </summary>
        public int WindowSize { get; set; } = 1024;

        /// <summary>
        /// Sampling rate of the audio signal in Hz
        /// </summary>
        public int SamplingRate { get; set; } = 48000;

        /// <summary>
        /// Minimum frequency for Mel bands in Hz
        /// </summary>
        public int MinimumFrequency { get; set; } = 0;

        /// <summary>
        /// Maximum frequency for Mel bands in Hz
        /// </summary>
        public int MaximumFrequency { get; set; } = 8000;

        /// <summary>
        /// Whether to center the signal when applying STFT
        /// </summary>
        public bool CenterSignal { get; set; } = false;
    }

    /// <summary>
    /// Provides utility methods for audio processing, including STFT/ISTFT, window functions, 
    /// spectrum manipulation, tensor conversion, and signal normalization.
    /// </summary>
    public static class AudioProcessing
    {
        #region STFT Operations
        /// <summary>
        /// Computes the Short-Time Fourier Transform (STFT) of an audio signal
        /// </summary>
        /// <param name="signal">Input audio signal (1D float array)</param>
        /// <param name="parameters">STFT configuration parameters</param>
        /// <param name="center">Whether to center the signal before processing</param>
        /// <param name="periodic">Whether the window function is periodic</param>
        /// <param name="onesided">Whether to return a one-sided spectrum (only positive frequencies)</param>
        /// <param name="normalized">Whether to normalize the STFT output</param>
        /// <param name="paddingMode">Padding mode for signal extension (e.g., "reflect", "constant")</param>
        /// <returns>3D array of complex STFT coefficients with shape [frequency_bins, channels, time_frames]</returns>
        /// <exception cref="ArgumentException">Thrown when input signal is null or empty</exception>
        /// <exception cref="InvalidOperationException">Thrown when window type is unsupported</exception>
        public static Complex[,,] ComputeStft(
            float[] signal,
            StftParameters parameters,
            bool center = false,
            bool periodic = false,
            bool? onesided = false,
            bool normalized = false,
            string paddingMode = "reflect")
        {
            if (signal == null || signal.Length == 0)
                throw new ArgumentException("Input audio signal cannot be null or empty.", nameof(signal));

            var window = CreateWindow(parameters.WindowType, parameters.WindowLength, periodic);
            if (window == null)
                throw new InvalidOperationException($"Unsupported window type: '{parameters.WindowType}'. Supported types: 'hamming', 'hanning'");

            return STFTFastWithMathNetNumerics.ComputeSTFT(
                input: signal,
                nFft: parameters.FftLength,
                hopLength: parameters.WindowIncrement,
                winLength: parameters.WindowLength,
                center: center,
                window: window,
                normalized: normalized,
                padMode: paddingMode,
                onesided: onesided ?? true);
        }

        /// <summary>
        /// Converts STFT format from float[1, 2*frequency_bins, time_frames] to Complex[frequency_bins, 2, time_frames]
        /// </summary>
        /// <param name="stftData">STFT array with shape [1, 2*frequency_bins, time_frames]</param>
        /// <returns>Complex spectrogram with shape [frequency_bins, 2, time_frames]</returns>
        /// <exception cref="ArgumentNullException">Thrown when input stftData is null</exception>
        public static Complex[,,] ConvertStftFormatToComplex(float[,,] stftData)
        {
            if (stftData == null)
                throw new ArgumentNullException(nameof(stftData));

            int frequencyBins = stftData.GetLength(1) / 2;
            int timeFrames = stftData.GetLength(2);
            const int channels = 2;

            var complexSpectrogram = new Complex[frequencyBins, channels, timeFrames];

            for (int f = 0; f < frequencyBins; f++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        int sourceIndex = c == 0 ? f : f + frequencyBins;
                        complexSpectrogram[f, c, t] = new Complex(stftData[0, sourceIndex, t], 0);
                    }
                }
            }

            return complexSpectrogram;
        }
        #endregion

        #region ISTFT Operations
        /// <summary>
        /// Computes the Inverse Short-Time Fourier Transform (ISTFT) from a 2D complex spectrogram
        /// </summary>
        /// <param name="spectrogram">2D complex spectrogram [frequency_bins, time_frames]</param>
        /// <param name="parameters">STFT configuration parameters (used for inverse transform)</param>
        /// <param name="targetLength">Desired output signal length (optional)</param>
        /// <param name="center">Whether the signal was centered during STFT</param>
        /// <param name="normalized">Whether the STFT was normalized</param>
        /// <param name="periodic">Whether the window function is periodic</param>
        /// <param name="onesided">Whether the input is a one-sided spectrum</param>
        /// <param name="returnComplex">Whether to return complex output (not supported for real signals)</param>
        /// <param name="window">Window function (auto-created if null)</param>
        /// <returns>Reconstructed audio signal</returns>
        /// <exception cref="ArgumentNullException">Thrown when input spectrogram is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when window type is unsupported</exception>
        public static float[] ComputeIstft(
            Complex[,] spectrogram,
            StftParameters parameters,
            int? targetLength = null,
            bool center = false,
            bool normalized = false,
            bool periodic = false,
            bool? onesided = null,
            bool returnComplex = false,
            float[] window = null)
        {
            if (spectrogram == null)
                throw new ArgumentNullException(nameof(spectrogram));

            window ??= CreateWindow(parameters.WindowType, parameters.WindowLength, periodic);
            if (window == null)
                throw new InvalidOperationException($"Unsupported window type: '{parameters.WindowType}'. Supported types: 'hamming', 'hanning'");

            return ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input2D: spectrogram,
                nFft: parameters.FftLength,
                hopLength: parameters.WindowIncrement,
                winLength: parameters.WindowLength,
                window: window,
                center: center,
                normalized: normalized,
                onesided: onesided ?? true,
                length: targetLength,
                returnComplex: returnComplex);
        }

        /// <summary>
        /// Computes the Inverse Short-Time Fourier Transform (ISTFT) from a 3D complex spectrogram
        /// </summary>
        /// <param name="spectrogram">3D complex spectrogram [frequency_bins, channels, time_frames]</param>
        /// <param name="parameters">STFT configuration parameters (used for inverse transform)</param>
        /// <param name="targetLength">Desired output signal length (optional)</param>
        /// <param name="center">Whether the signal was centered during STFT</param>
        /// <param name="normalized">Whether the STFT was normalized</param>
        /// <param name="periodic">Whether the window function is periodic</param>
        /// <param name="onesided">Whether the input is a one-sided spectrum</param>
        /// <param name="returnComplex">Whether to return complex output (not supported for real signals)</param>
        /// <param name="window">Window function (auto-created if null)</param>
        /// <returns>Reconstructed audio signal</returns>
        /// <exception cref="ArgumentNullException">Thrown when input spectrogram is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when window type is unsupported</exception>
        public static float[] ComputeIstft(
            Complex[,,] spectrogram,
            StftParameters parameters,
            int? targetLength = null,
            bool center = false,
            bool normalized = false,
            bool periodic = false,
            bool? onesided = null,
            bool returnComplex = false,
            float[] window = null)
        {
            if (spectrogram == null)
                throw new ArgumentNullException(nameof(spectrogram));

            window ??= CreateWindow(parameters.WindowType, parameters.WindowLength, periodic);
            if (window == null)
                throw new InvalidOperationException($"Unsupported window type: '{parameters.WindowType}'. Supported types: 'hamming', 'hanning'");

            return ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input: spectrogram,
                nFft: parameters.FftLength,
                hopLength: parameters.WindowIncrement,
                winLength: parameters.WindowLength,
                window: window,
                center: center,
                normalized: normalized,
                onesided: onesided ?? true,
                length: targetLength,
                returnComplex: returnComplex);
        }
        #endregion

        #region Window Functions
        /// <summary>
        /// Creates a window function of the specified type and length
        /// </summary>
        /// <param name="windowType">Type of window function ("hamming" or "hanning")</param>
        /// <param name="windowLength">Length of the window in samples</param>
        /// <param name="periodic">Whether the window is periodic (adds extra sample for FFT)</param>
        /// <returns>Window function array of length windowLength</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when windowLength is non-positive</exception>
        private static float[] CreateWindow(string windowType, int windowLength, bool periodic)
        {
            if (windowLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(windowLength), "Window length must be a positive integer.");

            var window = new float[windowLength];
            int adjustedLength = periodic ? windowLength : windowLength - 1;

            if (string.Equals(windowType, "hamming", StringComparison.OrdinalIgnoreCase))
            {
                for (int i = 0; i < windowLength; i++)
                {
                    window[i] = 0.54f - 0.46f * (float)Math.Cos(2 * Math.PI * i / adjustedLength);
                }
            }
            else if (string.Equals(windowType, "hanning", StringComparison.OrdinalIgnoreCase))
            {
                for (int i = 0; i < windowLength; i++)
                {
                    window[i] = 0.5f * (1 - (float)Math.Cos(2 * Math.PI * i / adjustedLength));
                }
            }
            else
            {
                return null;
            }

            return window;
        }
        #endregion

        #region Array Manipulations
        /// <summary>
        /// Repeats a 1D window array into a 3D array with shape [1, windowLength, repeatCount]
        /// </summary>
        /// <param name="window">Original 1D window array</param>
        /// <param name="repeatCount">Number of repetitions in the third dimension</param>
        /// <returns>3D array with shape [1, windowLength, repeatCount]</returns>
        /// <exception cref="ArgumentNullException">Thrown when window is null</exception>
        /// <exception cref="ArgumentException">Thrown when window is empty</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when repeatCount is non-positive</exception>
        public static float[,,] RepeatTo3DArray(float[] window, int repeatCount)
        {
            if (window == null)
                throw new ArgumentNullException(nameof(window));
            if (window.Length == 0)
                throw new ArgumentException("Window array cannot be empty.", nameof(window));
            if (repeatCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(repeatCount), "Repeat count must be a positive integer.");

            int windowLength = window.Length;
            var result = new float[1, windowLength, repeatCount];

            for (int d = 0; d < repeatCount; d++)
            {
                for (int i = 0; i < windowLength; i++)
                {
                    result[0, i, d] = window[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Permutes dimensions of a 2D float array
        /// </summary>
        /// <param name="tensor">Input 2D array</param>
        /// <param name="dim0">New first dimension index (0 or 1)</param>
        /// <param name="dim1">New second dimension index (0 or 1)</param>
        /// <returns>2D array with permuted dimensions</returns>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
        /// <exception cref="ArgumentException">Thrown when dimension indices are invalid</exception>
        public static float[,] PermuteDimensions(float[,] tensor, int dim0, int dim1)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (dim0 is < 0 or > 1 || dim1 is < 0 or > 1)
                throw new ArgumentException("Dimension indices must be 0 or 1 for 2D arrays.", nameof(dim0));

            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);
            var result = new float[size0, size1];

            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    int newI = dim0 == 0 ? i : j;
                    int newJ = dim1 == 0 ? i : j;
                    result[newI, newJ] = tensor[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Permutes dimensions of a 3D float array
        /// </summary>
        /// <param name="tensor">Input 3D array</param>
        /// <param name="dim0">New first dimension index (0, 1, or 2)</param>
        /// <param name="dim1">New second dimension index (0, 1, or 2)</param>
        /// <param name="dim2">New third dimension index (0, 1, or 2)</param>
        /// <returns>3D array with permuted dimensions</returns>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
        /// <exception cref="ArgumentException">Thrown when dimension indices are invalid</exception>
        public static float[,,] PermuteDimensions(float[,,] tensor, int dim0, int dim1, int dim2)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            ValidateDimensionIndices(dim0, dim1, dim2);

            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);
            int size2 = tensor.GetLength(dim2);
            var result = new float[size0, size1, size2];

            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    for (int k = 0; k < tensor.GetLength(2); k++)
                    {
                        int newI = GetNewIndex(dim0, i, j, k);
                        int newJ = GetNewIndex(dim1, i, j, k);
                        int newK = GetNewIndex(dim2, i, j, k);
                        result[newI, newJ, newK] = tensor[i, j, k];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Converts and permutes dimensions of a 3D complex array to a float array (using real parts)
        /// </summary>
        /// <param name="tensor">Input 3D complex array</param>
        /// <param name="dim0">New first dimension index (0, 1, or 2)</param>
        /// <param name="dim1">New second dimension index (0, 1, or 2)</param>
        /// <param name="dim2">New third dimension index (0, 1, or 2)</param>
        /// <returns>3D float array with permuted dimensions (contains real parts of complex numbers)</returns>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
        /// <exception cref="ArgumentException">Thrown when dimension indices are invalid</exception>
        public static float[,,] PermuteDimensions(Complex[,,] tensor, int dim0, int dim1, int dim2)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            ValidateDimensionIndices(dim0, dim1, dim2);

            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);
            int size2 = tensor.GetLength(dim2);
            var result = new float[size0, size1, size2];

            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    for (int k = 0; k < tensor.GetLength(2); k++)
                    {
                        int newI = GetNewIndex(dim0, i, j, k);
                        int newJ = GetNewIndex(dim1, i, j, k);
                        int newK = GetNewIndex(dim2, i, j, k);
                        result[newI, newJ, newK] = (float)tensor[i, j, k].Real;
                    }
                }
            }

            return result;
        }
        #endregion

        #region Spectrum Processing
        /// <summary>
        /// Applies a mask to a complex spectrum (real and imaginary parts)
        /// </summary>
        /// <param name="spectrum">Complex spectrum with shape [time_bins, frequency_bins, 2] 
        /// where the third dimension contains [real_part, imaginary_part]</param>
        /// <param name="mask">Mask array with shape [time_bins, frequency_bins, 1]</param>
        /// <returns>Masked complex spectrum with the same shape as input</returns>
        /// <exception cref="ArgumentNullException">Thrown when spectrum or mask is null</exception>
        /// <exception cref="ArgumentException">Thrown when mask dimensions do not match spectrum</exception>
        public static float[,,] ApplyMask(float[,,] spectrum, float[,,] mask)
        {
            if (spectrum == null)
                throw new ArgumentNullException(nameof(spectrum));
            if (mask == null)
                throw new ArgumentNullException(nameof(mask));

            int timeBins = spectrum.GetLength(0);
            int freqBins = spectrum.GetLength(1);

            if (mask.GetLength(0) != timeBins || mask.GetLength(1) != freqBins || mask.GetLength(2) != 1)
                throw new ArgumentException("Mask dimensions do not match spectrum dimensions. " +
                                          $"Expected [timeBins={timeBins}, freqBins={freqBins}, 1], " +
                                          $"got [{mask.GetLength(0)}, {mask.GetLength(1)}, {mask.GetLength(2)}].");

            var result = new float[timeBins, freqBins, 2];

            for (int t = 0; t < timeBins; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    float maskValue = mask[t, f, 0];
                    result[t, f, 0] = spectrum[t, f, 0] * maskValue;  // Apply to real part
                    result[t, f, 1] = spectrum[t, f, 1] * maskValue;  // Apply to imaginary part
                }
            }

            return result;
        }

        /// <summary>
        /// Converts a float[,,] complex spectrum (with real/imaginary parts) to a Complex[,] array
        /// </summary>
        /// <param name="spectrum">3D array with shape [time_bins, frequency_bins, 2] 
        /// where the third dimension contains [real_part, imaginary_part]</param>
        /// <returns>2D complex array with shape [time_bins, frequency_bins]</returns>
        /// <exception cref="ArgumentNullException">Thrown when spectrum is null</exception>
        /// <exception cref="ArgumentException">Thrown when spectrum's third dimension is not 2</exception>
        public static Complex[,] ConvertToComplex(float[,,] spectrum)
        {
            if (spectrum == null)
                throw new ArgumentNullException(nameof(spectrum));
            if (spectrum.GetLength(2) != 2)
                throw new ArgumentException("Spectrum must have 2 components (real/imaginary) in the third dimension.", nameof(spectrum));

            int timeBins = spectrum.GetLength(0);
            int freqBins = spectrum.GetLength(1);
            var complexSpectrum = new Complex[timeBins, freqBins];

            for (int t = 0; t < timeBins; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    complexSpectrum[t, f] = new Complex(spectrum[t, f, 0], spectrum[t, f, 1]);
                }
            }

            return complexSpectrum;
        }
        #endregion

        #region Tensor Conversions
        /// <summary>
        /// Converts a 3D Tensor<float> to a 3D float array
        /// </summary>
        /// <param name="tensor">Input 3D tensor</param>
        /// <returns>3D float array with the same dimensions as the input tensor</returns>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
        /// <exception cref="ArgumentException">Thrown when tensor is not 3-dimensional</exception>
        public static float[,,] TensorTo3DArray(Tensor<float> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional.", nameof(tensor));

            int dim0 = tensor.Dimensions[0];
            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];
            var array3D = new float[dim0, dim1, dim2];

            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        array3D[i, j, k] = tensor[i, j, k];
                    }
                }
            }

            return array3D;
        }

        /// <summary>
        /// Converts a 3D DenseTensor<float> to a 3D float array (optimized for dense storage)
        /// </summary>
        /// <param name="tensor">Input 3D dense tensor</param>
        /// <returns>3D float array with the same dimensions as the input tensor</returns>
        /// <exception cref="ArgumentNullException">Thrown when tensor is null</exception>
        /// <exception cref="ArgumentException">Thrown when tensor is not 3-dimensional</exception>
        public static float[,,] TensorTo3DArray(DenseTensor<float> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional.", nameof(tensor));

            int dim0 = tensor.Dimensions[0];
            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];
            var array3D = new float[dim0, dim1, dim2];
            var tensorSpan = tensor.Buffer.Span;

            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        int linearIndex = i * dim1 * dim2 + j * dim2 + k;
                        array3D[i, j, k] = tensorSpan[linearIndex];
                    }
                }
            }

            return array3D;
        }
        #endregion

        #region Signal Normalization
        /// <summary>
        /// Calculates the Root Mean Square (RMS) of a signal
        /// </summary>
        /// <param name="signal">Input signal array</param>
        /// <returns>RMS value of the signal</returns>
        /// <exception cref="ArgumentException">Thrown when signal is null or empty</exception>
        public static float CalculateRms(float[] signal)
        {
            if (signal == null || signal.Length == 0)
                throw new ArgumentException("Input signal cannot be null or empty.", nameof(signal));

            double sumOfSquares = 0;
            foreach (float value in signal)
            {
                sumOfSquares += value * value;
            }

            return (float)Math.Sqrt(sumOfSquares / signal.Length);
        }

        /// <summary>
        /// Normalizes a sample to match a target RMS value
        /// </summary>
        /// <param name="sample">Input audio sample</param>
        /// <param name="targetRms">Target RMS value (uses sample's RMS if null)</param>
        /// <returns>Normalized sample with the specified RMS</returns>
        /// <exception cref="ArgumentException">Thrown when sample is null or empty</exception>
        public static float[] NormalizeSample(float[] sample, float? targetRms = null)
        {
            if (sample == null || sample.Length == 0)
                throw new ArgumentException("Sample cannot be null or empty.", nameof(sample));

            float effectiveTargetRms = targetRms ?? CalculateRms(sample);
            float sampleRms = CalculateRms(sample);

            // Avoid division by zero with a small epsilon
            const float epsilon = 1e-10f;
            sampleRms = Math.Max(sampleRms, epsilon);

            var normalizedSample = new float[sample.Length];
            for (int i = 0; i < sample.Length; i++)
            {
                normalizedSample[i] = sample[i] / sampleRms * effectiveTargetRms;
            }

            return normalizedSample;
        }
        #endregion

        #region Helper Methods
        /// <summary>
        /// Validates that dimension indices are within the valid range (0-2) for 3D arrays
        /// </summary>
        /// <param name="dim0">First dimension index</param>
        /// <param name="dim1">Second dimension index</param>
        /// <param name="dim2">Third dimension index</param>
        /// <exception cref="ArgumentException">Thrown when any index is outside 0-2</exception>
        private static void ValidateDimensionIndices(int dim0, int dim1, int dim2)
        {
            if (dim0 is < 0 or > 2 || dim1 is < 0 or > 2 || dim2 is < 0 or > 2)
                throw new ArgumentException("Dimension indices must be 0, 1, or 2 for 3D arrays.");
        }

        /// <summary>
        /// Gets the new index based on dimension permutation for 3D arrays
        /// </summary>
        /// <param name="dimension">Target dimension (0, 1, or 2)</param>
        /// <param name="i">Original first dimension index</param>
        /// <param name="j">Original second dimension index</param>
        /// <param name="k">Original third dimension index</param>
        /// <returns>New index corresponding to the target dimension</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when dimension is invalid</exception>
        private static int GetNewIndex(int dimension, int i, int j, int k) => dimension switch
        {
            0 => i,
            1 => j,
            2 => k,
            _ => throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be 0, 1, or 2.")
        };
        #endregion
    }
}