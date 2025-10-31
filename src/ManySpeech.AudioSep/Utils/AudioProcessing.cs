using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Numerics;

namespace ManySpeech.AudioSep.Utils
{
    /// <summary>
    /// Configuration arguments for Short-Time Fourier Transform (STFT)
    /// </summary>
    public class STFTArgs
    {
        public string WinType { get; set; }
        public int WinLen { get; set; }
        public int WinInc { get; set; }
        public int FftLen { get; set; }
    }

    /// <summary>
    /// Configuration arguments for Mel spectrogram generation
    /// </summary>
    public class MelArgs
    {
        public int NFFT { get; set; } = 1024;
        public int NumMels { get; set; } = 80;
        public int HopSize { get; set; } = 256;
        public int WinSize { get; set; } = 1024;
        public int SamplingRate { get; set; } = 48000;
        public int Fmin { get; set; } = 0;
        public int Fmax { get; set; } = 8000;
        public bool Center { get; set; } = false;
    }

    /// <summary>
    /// Provides audio processing utilities including STFT, ISTFT, window functions, and spectrum manipulations
    /// </summary>
    public static class AudioProcessing
    {
        #region STFT Operations
        /// <summary>
        /// Computes the Short-Time Fourier Transform (STFT)
        /// </summary>
        /// <param name="x">Input audio signal</param>
        /// <param name="args">STFT configuration parameters</param>
        /// <param name="center">Whether to center the signal</param>
        /// <param name="periodic">Whether the window is periodic</param>
        /// <param name="onesided">Whether to return one-sided spectrum</param>
        /// <param name="normalized">Whether to normalize the STFT</param>
        /// <param name="padMode">Padding mode for signal extension</param>
        /// <returns>3D array of complex STFT coefficients with shape [freq_bins, channels, time_frames]</returns>
        public static Complex[,,] Stft(float[] x, STFTArgs args, bool center = false, bool periodic = false,
                                      bool? onesided = false, bool normalized = false, string padMode = "reflect")
        {
            if (x == null || x.Length == 0)
                throw new ArgumentException("Input audio signal cannot be null or empty", nameof(x));

            var window = CreateWindow(args.WinType, args.WinLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In STFT, window type '{args.WinType}' is not supported!");
                return null;
            }

            return STFTFastWithMathNetNumerics.ComputeSTFT(
                input: x,
                n_fft: args.FftLen,
                hop_length: args.WinInc,
                win_length: args.WinLen,
                center: center,
                window: window,
                normalized: normalized,
                pad_mode: padMode,
                onesided: true);
        }

        /// <summary>
        /// Converts STFT format from float[1, 2*freq_bins, time_frames] to Complex[321, 2, 723]
        /// </summary>
        /// <param name="stftFormat">STFT array with shape [1, 2*freq_bins, time_frames]</param>
        /// <returns>Complex spectrogram with shape [freq_bins, 1, time_frames]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex2(float[,,] stftFormat)
        {
            if (stftFormat == null)
                throw new ArgumentNullException(nameof(stftFormat));

            int freqBins = stftFormat.GetLength(1) / 2;
            int timeFrames = stftFormat.GetLength(2);
            int channels = 2;

            var complexSpectrogram = new Complex[freqBins, channels, timeFrames];

            for (int f = 0; f < freqBins; f++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        int sourceIndex = c == 0 ? f : f + freqBins;
                        complexSpectrogram[f, c, t] = new Complex(stftFormat[0, sourceIndex, t], 0);
                    }
                }
            }

            return complexSpectrogram;
        }
        #endregion

        #region ISTFT Operations
        /// <summary>
        /// Computes the Inverse Short-Time Fourier Transform (ISTFT) for 2D complex input
        /// </summary>
        /// <param name="x">2D complex spectrogram</param>
        /// <param name="args">STFT configuration parameters (used for inverse transform)</param>
        /// <param name="slen">Desired output signal length</param>
        /// <param name="center">Whether the signal was centered during STFT</param>
        /// <param name="normalized">Whether the STFT was normalized</param>
        /// <param name="periodic">Whether the window is periodic</param>
        /// <param name="onesided">Whether the input is one-sided spectrum</param>
        /// <param name="returnComplex">Whether to return complex output (not used for real signals)</param>
        /// <param name="window">Window function (auto-created if null)</param>
        /// <returns>Reconstructed audio signal</returns>
        public static float[] Istft(Complex[,] x, STFTArgs args, int? slen = null, bool center = false,
                                   bool normalized = false, bool periodic = false, bool? onesided = null,
                                   bool returnComplex = false, float[] window = null)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            window ??= CreateWindow(args.WinType, args.WinLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In ISTFT, window type '{args.WinType}' is not supported!");
                return null;
            }

            return ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input2D: x,
                n_fft: args.FftLen,
                hop_length: args.WinInc,
                win_length: args.WinLen,
                window: window,
                center: true,
                normalized: normalized,
                onesided: true,
                length: slen
            );
        }

        /// <summary>
        /// Computes the Inverse Short-Time Fourier Transform (ISTFT) for 3D complex input
        /// </summary>
        /// <param name="x">3D complex spectrogram</param>
        /// <param name="args">STFT configuration parameters (used for inverse transform)</param>
        /// <param name="slen">Desired output signal length</param>
        /// <param name="center">Whether the signal was centered during STFT</param>
        /// <param name="normalized">Whether the STFT was normalized</param>
        /// <param name="periodic">Whether the window is periodic</param>
        /// <param name="onesided">Whether the input is one-sided spectrum</param>
        /// <param name="returnComplex">Whether to return complex output (not used for real signals)</param>
        /// <param name="window">Window function (auto-created if null)</param>
        /// <returns>Reconstructed audio signal</returns>
        public static float[] Istft(Complex[,,] x, STFTArgs args, int? slen = null, bool center = false,
                                   bool normalized = false, bool periodic = false, bool? onesided = null,
                                   bool returnComplex = false, float[] window = null)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            window ??= CreateWindow(args.WinType, args.WinLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In ISTFT, window type '{args.WinType}' is not supported!");
                return null;
            }

            return ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input: x,
                n_fft: args.FftLen,
                hop_length: args.WinInc,
                win_length: args.WinLen,
                window: window,
                center: true,
                normalized: normalized,
                onesided: true,
                length: slen
            );
        }
        #endregion

        #region Window Functions
        /// <summary>
        /// Creates a window function of specified type and length
        /// </summary>
        /// <param name="winType">Type of window (hamming or hanning)</param>
        /// <param name="winLen">Length of the window</param>
        /// <param name="periodic">Whether the window is periodic</param>
        /// <returns>Window function array</returns>
        private static float[] CreateWindow(string winType, int winLen, bool periodic)
        {
            if (winLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(winLen), "Window length must be positive");

            var window = new float[winLen];
            int adjustedLength = periodic ? winLen : winLen - 1;

            if (string.Equals(winType, "hamming", StringComparison.OrdinalIgnoreCase))
            {
                for (int i = 0; i < winLen; i++)
                {
                    window[i] = 0.54f - 0.46f * (float)Math.Cos(2 * Math.PI * i / adjustedLength);
                }
            }
            else if (string.Equals(winType, "hanning", StringComparison.OrdinalIgnoreCase))
            {
                for (int i = 0; i < winLen; i++)
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
        /// Repeats a 1D array into a 3D array with shape [1, windowLength, repeatCount]
        /// </summary>
        /// <param name="window">Original 1D array with length windowLength</param>
        /// <param name="repeatCount">Number of repetitions in the third dimension</param>
        /// <returns>3D array with shape [1, windowLength, repeatCount]</returns>
        public static float[,,] RepeatTo3DArray(float[] window, int repeatCount)
        {
            if (window == null)
                throw new ArgumentNullException(nameof(window));
            if (window.Length == 0)
                throw new ArgumentException("Window array cannot be empty", nameof(window));
            if (repeatCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(repeatCount), "Repeat count must be positive");

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
        public static float[,] PermuteDimensions(float[,] tensor, int dim0, int dim1)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (dim0 < 0 || dim0 > 1 || dim1 < 0 || dim1 > 1)
                throw new ArgumentException("Dimension indices must be 0 or 1", nameof(dim0));

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
        /// Converts and permutes dimensions of a 3D complex array to float array (using real parts)
        /// </summary>
        /// <param name="tensor">Input 3D complex array</param>
        /// <param name="dim0">New first dimension index (0, 1, or 2)</param>
        /// <param name="dim1">New second dimension index (0, 1, or 2)</param>
        /// <param name="dim2">New third dimension index (0, 1, or 2)</param>
        /// <returns>3D float array with permuted dimensions (real parts only)</returns>
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
        /// Applies a mask to a complex spectrum
        /// </summary>
        /// <param name="spectrum">Complex spectrum with shape [time, freq, 2] (real/imaginary parts)</param>
        /// <param name="mask">Mask array with shape [time, freq, 1]</param>
        /// <returns>Masked complex spectrum</returns>
        public static float[,,] ApplyMask(float[,,] spectrum, float[,,] mask)
        {
            if (spectrum == null)
                throw new ArgumentNullException(nameof(spectrum));
            if (mask == null)
                throw new ArgumentNullException(nameof(mask));

            int timeBins = spectrum.GetLength(0);
            int freqBins = spectrum.GetLength(1);

            // Validate mask dimensions
            if (mask.GetLength(0) != timeBins || mask.GetLength(1) != freqBins || mask.GetLength(2) != 1)
            {
                throw new ArgumentException("Mask dimensions do not match spectrum dimensions");
            }

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
        /// Converts a float[,,] complex spectrum to Complex[,] format
        /// </summary>
        /// <param name="spec">3D array with shape [time, freq, 2] (real/imaginary parts)</param>
        /// <returns>2D complex array with shape [time, freq]</returns>
        public static Complex[,] ConvertToComplex(float[,,] spec)
        {
            if (spec == null)
                throw new ArgumentNullException(nameof(spec));
            if (spec.GetLength(2) != 2)
                throw new ArgumentException("Spectrum must have 2 components (real/imaginary) in third dimension", nameof(spec));

            int timeBins = spec.GetLength(0);
            int freqBins = spec.GetLength(1);
            var complexSpec = new Complex[timeBins, freqBins];

            for (int t = 0; t < timeBins; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    complexSpec[t, f] = new Complex(spec[t, f, 0], spec[t, f, 1]);
                }
            }

            return complexSpec;
        }
        #endregion

        #region Tensor Conversions
        /// <summary>
        /// Converts a 3D Tensor<float> to a 3D float array
        /// </summary>
        /// <param name="tensor">Input 3D tensor</param>
        /// <returns>3D float array with the same dimensions</returns>
        public static float[,,] TensorTo3DArray(Tensor<float> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional", nameof(tensor));

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
        /// <returns>3D float array with the same dimensions</returns>
        public static float[,,] TensorTo3DArray(DenseTensor<float> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional", nameof(tensor));

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
        /// <param name="data">Input signal</param>
        /// <returns>RMS value</returns>
        public static float CalculateRms(float[] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Input data cannot be null or empty", nameof(data));

            double sumSquared = 0;
            foreach (var value in data)
            {
                sumSquared += value * value;
            }

            return (float)Math.Sqrt(sumSquared / data.Length);
        }

        /// <summary>
        /// Normalizes a sample to match a target RMS value
        /// </summary>
        /// <param name="sample">Input audio sample</param>
        /// <param name="rmsInput">Target RMS value (uses sample's RMS if null)</param>
        /// <returns>Normalized sample</returns>
        public static float[] NormalizeSample(float[] sample, float? rmsInput = null)
        {
            if (sample == null || sample.Length == 0)
                throw new ArgumentException("Sample cannot be null or empty", nameof(sample));

            float effectiveRmsInput = rmsInput ?? CalculateRms(sample);
            float rmsOut = CalculateRms(sample);

            // Avoid division by zero
            if (rmsOut < 1e-10f)
            {
                rmsOut = 1e-10f;
            }

            var result = new float[sample.Length];
            for (int i = 0; i < sample.Length; i++)
            {
                result[i] = sample[i] / rmsOut * effectiveRmsInput;
            }

            return result;
        }
        #endregion

        #region Helper Methods
        /// <summary>
        /// Validates that dimension indices are within valid range (0-2)
        /// </summary>
        private static void ValidateDimensionIndices(int dim0, int dim1, int dim2)
        {
            if (dim0 < 0 || dim0 > 2 || dim1 < 0 || dim1 > 2 || dim2 < 0 || dim2 > 2)
                throw new ArgumentException("Dimension indices must be 0, 1, or 2");
        }

        /// <summary>
        /// Gets new index based on dimension permutation
        /// </summary>
        private static int GetNewIndex(int dim, int i, int j, int k) => dim switch
        {
            0 => i,
            1 => j,
            2 => k,
            _ => throw new ArgumentOutOfRangeException(nameof(dim))
        };
        #endregion
    }
}