using System;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex = System.Numerics.Complex;

namespace ManySpeech.AudioSep.Utils
{
    /// <summary>
    /// Provides fast Short-Time Fourier Transform (STFT) implementation using MathNet.Numerics
    /// </summary>
    public class STFTFastWithMathNetNumerics
    {
        /// <summary>
        /// Computes the Short-Time Fourier Transform (STFT)
        /// </summary>
        /// <param name="input">Input audio signal</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="hopLength">Hop length between consecutive frames (defaults to nFft/4)</param>
        /// <param name="winLength">Window length (defaults to nFft)</param>
        /// <param name="window">Window function (auto-generated if null)</param>
        /// <param name="center">Whether to center the signal before processing</param>
        /// <param name="padMode">Padding mode for signal extension (supports: reflect, constant, replicate, circular)</param>
        /// <param name="normalized">Whether to normalize the STFT output</param>
        /// <param name="onesided">Whether to return one-sided spectrum (only positive frequencies)</param>
        /// <param name="returnComplex">Whether to return complex output (always true for this implementation)</param>
        /// <returns>3D array of complex STFT coefficients with shape [frequency_bins, 1, time_frames]</returns>
        /// <exception cref="ArgumentException">Thrown for invalid input parameters</exception>
        public static Complex[,,] ComputeSTFT(
            float[] input,
            int nFft,
            int? hopLength = null,
            int? winLength = null,
            float[] window = null,
            bool center = true,
            string padMode = "reflect",
            bool normalized = false,
            bool? onesided = null,
            bool returnComplex = true)
        {
            // Parameter validation and default assignments
            hopLength ??= nFft / 4;
            winLength ??= nFft;
            onesided ??= true;

            ValidateParameters(input, nFft, hopLength.Value, winLength.Value);

            // Window handling (matches PyTorch behavior, no COLA normalization by default)
            window ??= CreateHannWindow(winLength.Value);
            if (window.Length < nFft)
            {
                window = PadWindowToSize(window, nFft);
            }

            // Apply normalization for COLA (Constant Overlap-Add) condition if needed
            if (normalized)
            {
                float windowEnergy = window.Sum(x => x * x);
                float normalizationFactor = (float)Math.Sqrt(nFft * windowEnergy / hopLength.Value);
                for (int i = 0; i < window.Length; i++)
                {
                    window[i] /= normalizationFactor;
                }
            }

            // Pad input if centering is enabled
            float[] processedInput = center ? PadInput(input, nFft, padMode) : input;

            // Calculate frame and frequency bin counts
            int frameCount = (processedInput.Length - nFft) / hopLength.Value + 1;
            int freqBinCount = onesided.Value ? (nFft / 2 + 1) : nFft;

            // Initialize output array [frequency_bins, 1, time_frames]
            var stftResult = new Complex[freqBinCount, 1, frameCount];

            // Process frames in parallel with thread-local buffer
            Parallel.For(
                0,
                frameCount,
                () => new Complex[nFft],  // Thread-local FFT buffer
                (frameIdx, loopState, frameComplex) =>
                {
                    // Extract frame and apply window function
                    int frameOffset = frameIdx * hopLength.Value;
                    for (int i = 0; i < nFft; i++)
                    {
                        float sample = (frameOffset + i < processedInput.Length) ? processedInput[frameOffset + i] : 0f;
                        frameComplex[i] = new Complex(sample * window[i], 0);
                    }

                    // Perform forward FFT without scaling
                    Fourier.Forward(frameComplex, FourierOptions.NoScaling);

                    // Apply normalization if enabled
                    if (normalized)
                    {
                        double scale = 1.0 / Math.Sqrt(nFft);
                        for (int i = 0; i < nFft; i++)
                        {
                            frameComplex[i] *= scale;
                        }
                    }

                    // Store results (one-sided or full spectrum)
                    if (onesided.Value)
                    {
                        for (int freqIdx = 0; freqIdx < freqBinCount; freqIdx++)
                        {
                            stftResult[freqIdx, 0, frameIdx] = frameComplex[freqIdx];
                        }
                    }
                    else
                    {
                        for (int freqIdx = 0; freqIdx < nFft; freqIdx++)
                        {
                            stftResult[freqIdx, 0, frameIdx] = frameComplex[freqIdx];
                        }
                    }

                    return frameComplex;
                },
                _ => { }  // No need to process thread-local buffers after use
            );

            return stftResult;
        }

        #region Validation Methods
        /// <summary>
        /// Validates STFT input parameters
        /// </summary>
        /// <param name="input">Input audio signal</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="hopLength">Hop length</param>
        /// <param name="winLength">Window length</param>
        /// <exception cref="ArgumentException">Thrown for invalid parameters</exception>
        private static void ValidateParameters(float[] input, int nFft, int hopLength, int winLength)
        {
            if (input == null || input.Length == 0)
                throw new ArgumentException("Input signal cannot be null or empty.", nameof(input));

            if (nFft <= 0)
                throw new ArgumentException("FFT size (nFft) must be a positive integer.", nameof(nFft));

            if (hopLength <= 0)
                throw new ArgumentException("Hop length must be a positive integer.", nameof(hopLength));

            if (winLength <= 0 || winLength > nFft)
                throw new ArgumentException("Window length must be positive and not exceed FFT size.", nameof(winLength));
        }
        #endregion

        #region Window Functions
        /// <summary>
        /// Creates a Hann window function (matches torch.hann_window behavior)
        /// </summary>
        /// <param name="length">Window length</param>
        /// <param name="periodic">Whether the window is periodic (adds extra sample for FFT)</param>
        /// <returns>Hann window array</returns>
        private static float[] CreateHannWindow(int length, bool periodic = true)
        {
            int adjustedLength = periodic ? length + 1 : length;
            float[] window = new float[adjustedLength];

            for (int i = 0; i < adjustedLength; i++)
            {
                double angle = 2.0 * Math.PI * i / (adjustedLength - 1);
                window[i] = 0.5f * (1.0f - (float)Math.Cos(angle));
            }

            return periodic ? window.Take(length).ToArray() : window;
        }

        /// <summary>
        /// Pads a window function to the target size
        /// </summary>
        /// <param name="window">Original window array</param>
        /// <param name="targetLength">Desired window length</param>
        /// <returns>Padded window array</returns>
        private static float[] PadWindowToSize(float[] window, int targetLength)
        {
            if (window.Length == targetLength)
                return window;

            float[] paddedWindow = new float[targetLength];
            int leftPad = (targetLength - window.Length) / 2;
            Array.Copy(window, 0, paddedWindow, leftPad, window.Length);

            return paddedWindow;
        }
        #endregion

        #region Input Padding
        /// <summary>
        /// Pads input signal according to specified mode (matches torch.stft padding behavior)
        /// </summary>
        /// <param name="input">Input signal to pad</param>
        /// <param name="nFft">FFT size (determines padding width)</param>
        /// <param name="mode">Padding mode (reflect, constant, replicate, circular)</param>
        /// <returns>Padded input signal</returns>
        /// <exception cref="ArgumentException">Thrown for unsupported padding modes</exception>
        private static float[] PadInput(float[] input, int nFft, string mode)
        {
            int padWidth = nFft / 2;
            float[] paddedInput = new float[input.Length + 2 * padWidth];

            // Apply padding based on mode
            switch (mode.ToLowerInvariant())
            {
                case "reflect":
                    ApplyReflectPadding(input, paddedInput, padWidth);
                    break;
                case "constant":
                    // Constant padding (defaults to 0, no action needed as array initializes to 0)
                    break;
                case "replicate":
                    ApplyReplicatePadding(input, paddedInput, padWidth);
                    break;
                case "circular":
                    ApplyCircularPadding(input, paddedInput, padWidth);
                    break;
                default:
                    throw new ArgumentException($"Unsupported padding mode: {mode}. Supported modes: reflect, constant, replicate, circular.", nameof(mode));
            }

            // Copy original signal to center of padded array
            Array.Copy(input, 0, paddedInput, padWidth, input.Length);

            return paddedInput;
        }

        /// <summary>
        /// Applies reflect padding (mirrors signal around edges)
        /// </summary>
        private static void ApplyReflectPadding(float[] input, float[] padded, int padWidth)
        {
            for (int i = 0; i < padWidth; i++)
            {
                padded[i] = input[padWidth - i - 1];
                padded[padded.Length - 1 - i] = input[input.Length - 1 - (padWidth - i - 1)];
            }
        }

        /// <summary>
        /// Applies replicate padding (extends edge values)
        /// </summary>
        private static void ApplyReplicatePadding(float[] input, float[] padded, int padWidth)
        {
            float firstValue = input[0];
            float lastValue = input[input.Length - 1];

            for (int i = 0; i < padWidth; i++)
            {
                padded[i] = firstValue;
                padded[padded.Length - 1 - i] = lastValue;
            }
        }

        /// <summary>
        /// Applies circular padding (wraps signal around)
        /// </summary>
        private static void ApplyCircularPadding(float[] input, float[] padded, int padWidth)
        {
            for (int i = 0; i < padWidth; i++)
            {
                padded[i] = input[input.Length - padWidth + i];
                padded[padded.Length - 1 - i] = input[padWidth - i - 1];
            }
        }
        #endregion
    }
}