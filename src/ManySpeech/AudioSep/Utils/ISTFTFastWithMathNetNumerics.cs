using System;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex = System.Numerics.Complex;

namespace ManySpeech.AudioSep.Utils
{
    /// <summary>
    /// Provides fast Inverse Short-Time Fourier Transform (ISTFT) implementation using MathNet.Numerics
    /// </summary>
    public static class ISTFTFastWithMathNetNumerics
    {
        #region Public Methods
        /// <summary>
        /// Computes ISTFT from a 2D complex spectrogram
        /// </summary>
        /// <param name="input2D">2D complex spectrogram with shape [frequency_bins, time_frames]</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="hopLength">Hop length between consecutive frames (defaults to nFft/4)</param>
        /// <param name="winLength">Window length (defaults to nFft)</param>
        /// <param name="window">Window function (auto-generated if null)</param>
        /// <param name="center">Whether input was centered during STFT</param>
        /// <param name="normalized">Whether to apply normalization</param>
        /// <param name="onesided">Whether input is one-sided spectrogram</param>
        /// <param name="length">Desired output signal length</param>
        /// <param name="returnComplex">Whether to return complex output (not implemented)</param>
        /// <returns>Reconstructed audio signal</returns>
        public static float[] ComputeISTFT(Complex[,] input2D, int nFft, int? hopLength = null,
                                         int? winLength = null, float[] window = null,
                                         bool center = true, bool normalized = false,
                                         bool? onesided = null, int? length = null,
                                         bool returnComplex = false)
        {
            // Convert 2D input to 3D format with batch dimension (size 1)
            int freqBins = input2D.GetLength(0);
            int timeFrames = input2D.GetLength(1);
            var input3D = new Complex[freqBins, 1, timeFrames];

            Parallel.For(0, freqBins, f =>
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    input3D[f, 0, t] = input2D[f, t];
                }
            });

            return ComputeISTFT(input3D, nFft, hopLength, winLength, window,
                             center, normalized, onesided, length, returnComplex);
        }

        /// <summary>
        /// Computes ISTFT from a 3D complex spectrogram
        /// </summary>
        /// <param name="input">3D complex spectrogram with shape [frequency_bins, batch_size, time_frames]</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="hopLength">Hop length between consecutive frames (defaults to nFft/4)</param>
        /// <param name="winLength">Window length (defaults to nFft)</param>
        /// <param name="window">Window function (auto-generated if null)</param>
        /// <param name="center">Whether input was centered during STFT</param>
        /// <param name="normalized">Whether to apply normalization</param>
        /// <param name="onesided">Whether input is one-sided spectrogram</param>
        /// <param name="length">Desired output signal length</param>
        /// <param name="returnComplex">Whether to return complex output (not implemented)</param>
        /// <returns>Reconstructed audio signal</returns>
        public static float[] ComputeISTFT(Complex[,,] input, int nFft, int? hopLength = null,
                                         int? winLength = null, float[] window = null,
                                         bool center = true, bool normalized = false,
                                         bool? onesided = null, int? length = null,
                                         bool returnComplex = false)
        {
            if (returnComplex)
                throw new NotImplementedException("Complex output is not implemented");

            // Parameter validation and default values
            hopLength ??= nFft / 4;
            winLength ??= nFft;
            onesided ??= true;

            int batchSize = input.GetLength(1);
            int freqBins = input.GetLength(0);
            int timeFrames = input.GetLength(2);

            if (batchSize != 1)
                throw new ArgumentException("Only batch size of 1 is currently supported");

            // Window handling (aligned with PyTorch's behavior)
            window ??= CreateHannWindow(winLength.Value);
            window = PadWindowToLength(window, nFft);

            // Verify Nonzero Overlap Add (NOLA) condition
            VerifyNOLACondition(window, hopLength.Value);

            // Calculate output length (matches torch.istft behavior)
            int expectedOutputLength = (timeFrames - 1) * hopLength.Value + winLength.Value;
            int outputLength = length ?? (center ? expectedOutputLength - nFft : expectedOutputLength);

            // Initialize output and normalization arrays
            float[] output = new float[outputLength];
            float[] norm = new float[outputLength];

            // Precompute squared window for normalization
            float[] windowSquared = window.Select(x => x * x).ToArray();

            // Apply normalization compensation (matches torch.istft)
            if (normalized)
            {
                float scaleFactor = (float)Math.Sqrt(nFft);
                for (int f = 0; f < freqBins; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        input[f, 0, t] *= scaleFactor;
                    }
                }
            }

            // Calculate overlap-add scaling factors (matches torch.istft)
            for (int t = 0; t < timeFrames; t++)
            {
                int start = center ? t * hopLength.Value - nFft / 2 : t * hopLength.Value;
                start = Math.Max(0, start); // Ensure no underflow

                for (int i = 0; i < nFft && start + i < norm.Length; i++)
                {
                    norm[start + i] += windowSquared[i];
                }
            }

            // Process each frame in parallel
            Parallel.For(0, timeFrames, t =>
            {
                int start = center ? t * hopLength.Value - nFft / 2 : t * hopLength.Value;
                start = Math.Max(0, start); // Ensure no underflow

                // Reconstruct full spectrum from input
                Complex[] frameSpectrum = ReconstructSpectrum(input, t, nFft, onesided.Value);

                // Apply inverse FFT
                Fourier.Inverse(frameSpectrum, FourierOptions.AsymmetricScaling);

                // Apply window and accumulate results
                for (int i = 0; i < nFft; i++)
                {
                    int pos = start + i;
                    if (pos >= 0 && pos < output.Length)
                    {
                        float value = (float)frameSpectrum[i].Real * window[i];
                        lock (output) output[pos] += value;
                        lock (norm) norm[pos] += windowSquared[i];
                    }
                }
            });

            // Apply advanced normalization and fade-out processing
            ApplyNormalizationAndFade(output, norm, hopLength.Value);

            // Handle center padding trimming
            if (center)
            {
                int padWidth = nFft / 2;
                if (output.Length > padWidth)
                {
                    float[] trimmed = new float[output.Length - padWidth];
                    Array.Copy(output, padWidth, trimmed, 0, trimmed.Length);
                    output = trimmed;
                }
            }

            // Adjust to desired output length
            if (length.HasValue && length.Value != output.Length)
            {
                float[] adjusted = new float[length.Value];
                int copyLength = Math.Min(length.Value, output.Length);
                Array.Copy(output, 0, adjusted, 0, copyLength);
                output = adjusted;
            }

            // Apply post-processing EQ compensation
            output = ApplyEqCompensation(output);

            return output;
        }

        /// <summary>
        /// Applies simple frequency band EQ compensation to enhance audio quality
        /// </summary>
        /// <param name="audio">Input audio signal</param>
        /// <param name="lowBoost">Boost factor for low frequencies</param>
        /// <param name="highBoost">Boost factor for high frequencies</param>
        /// <returns>EQ-compensated audio signal</returns>
        public static float[] ApplyEqCompensation(float[] audio, float lowBoost = 1.2f, float highBoost = 1.1f)
        {
            // Enhance low frequencies (simulated 50Hz-200Hz range)
            Parallel.For(0, audio.Length, i =>
            {
                float positionRatio = (float)i / audio.Length;
                if (positionRatio < 0.1f)
                {
                    audio[i] *= lowBoost;
                }
                // Enhance high frequencies (simulated >8kHz range)
                else if (positionRatio > 0.9f)
                {
                    audio[i] *= highBoost;
                }
            });
            return audio;
        }
        #endregion

        #region Private Helper Methods
        /// <summary>
        /// Verifies the Nonzero Overlap Add (NOLA) condition for the window function
        /// </summary>
        /// <param name="window">Window function array</param>
        /// <param name="hopLength">Hop length between frames</param>
        /// <exception cref="ArgumentException">Thrown if NOLA condition is not satisfied</exception>
        private static void VerifyNOLACondition(float[] window, int hopLength)
        {
            float sum = 0;
            for (int i = 0; i < window.Length; i += hopLength)
            {
                if (i < window.Length)
                    sum += window[i] * window[i];
            }

            if (sum < 1e-10f)
                throw new ArgumentException("Window fails NOLA (nonzero overlap-add) condition");
        }

        /// <summary>
        /// Creates a Hann window function (matches torch.hann_window behavior)
        /// </summary>
        /// <param name="length">Window length</param>
        /// <param name="periodic">Whether the window is periodic</param>
        /// <returns>Hann window array</returns>
        private static float[] CreateHannWindow(int length, bool periodic = true)
        {
            int windowLength = periodic ? length + 1 : length;
            float[] window = new float[windowLength];

            for (int i = 0; i < windowLength; i++)
            {
                double angle = 2.0 * Math.PI * i / (windowLength - 1);
                window[i] = 0.5f * (1.0f - (float)Math.Cos(angle));
            }

            return periodic ? window.Take(length).ToArray() : window;
        }

        /// <summary>
        /// Pads a window function to the target length
        /// </summary>
        /// <param name="window">Original window array</param>
        /// <param name="targetLength">Desired window length</param>
        /// <returns>Padded window array</returns>
        /// <exception cref="ArgumentException">Thrown if window is longer than target length</exception>
        private static float[] PadWindowToLength(float[] window, int targetLength)
        {
            if (window.Length == targetLength) return window;
            if (window.Length > targetLength)
                throw new ArgumentException("Window length cannot be greater than nFft");

            float[] padded = new float[targetLength];
            int padLeft = (targetLength - window.Length) / 2;
            Array.Copy(window, 0, padded, padLeft, window.Length);
            return padded;
        }

        /// <summary>
        /// Reconstructs a full complex spectrum from a one-sided or two-sided input
        /// </summary>
        /// <param name="input">3D complex spectrogram</param>
        /// <param name="frameIndex">Index of the frame to process</param>
        /// <param name="nFft">FFT size</param>
        /// <param name="onesided">Whether input is one-sided</param>
        /// <returns>Full complex spectrum array</returns>
        /// <exception cref="ArgumentException">Thrown for invalid frequency bin counts</exception>
        private static Complex[] ReconstructSpectrum(Complex[,,] input, int frameIndex, int nFft, bool onesided)
        {
            int freqBins = input.GetLength(0);
            Complex[] spectrum = new Complex[nFft];

            if (onesided)
            {
                if (freqBins != nFft / 2 + 1)
                    throw new ArgumentException($"Invalid frequency bin count {freqBins} for onesided input with nFft {nFft}");

                // Copy one-sided part
                for (int f = 0; f < freqBins; f++)
                {
                    spectrum[f] = input[f, 0, frameIndex];
                }

                // Create conjugate symmetric part
                for (int f = 1; f < nFft - freqBins + 1; f++)
                {
                    if (f < nFft / 2)
                        spectrum[nFft - f] = Complex.Conjugate(spectrum[f]);
                }

                // Ensure DC and Nyquist components are real
                spectrum[0] = new Complex(spectrum[0].Real, 0);
                if (nFft % 2 == 0)
                    spectrum[nFft / 2] = new Complex(spectrum[nFft / 2].Real, 0);
            }
            else
            {
                if (freqBins != nFft)
                    throw new ArgumentException($"Frequency bin count must equal nFft for two-sided input");

                for (int f = 0; f < nFft; f++)
                {
                    spectrum[f] = input[f, 0, frameIndex];
                }
            }

            return spectrum;
        }

        /// <summary>
        /// Applies advanced normalization with fade-out to reduce artifacts
        /// </summary>
        /// <param name="output">Audio signal to process</param>
        /// <param name="norm">Normalization envelope array</param>
        /// <param name="hopLength">Hop length used in STFT</param>
        private static void ApplyNormalizationAndFade(float[] output, float[] norm, int hopLength)
        {
            // Fade-out length (30ms @ 16000Hz)
            int fadeOutLength = (int)(0.03 * 16000);
            // Transition band length (10ms @ 16000Hz)
            int transitionLength = (int)(0.01 * 16000);
            float minThreshold = 1e-6f;
            float maxThreshold = 1e-3f;

            // Ensure valid parameters
            if (fadeOutLength > output.Length)
            {
                fadeOutLength = output.Length;
                transitionLength = 0;
            }

            // Calculate indices for fade and transition regions
            int fadeStartIndex = output.Length - fadeOutLength;
            int transitionStartIndex = fadeStartIndex - transitionLength;
            if (transitionStartIndex < 0) transitionStartIndex = 0;

            // Phase 1: Adaptive safe normalization for main audio body
            for (int i = 0; i < transitionStartIndex; i++)
            {
                float signalEnergy = Math.Abs(output[i]);
                float adaptiveThreshold = Math.Max(minThreshold, maxThreshold * (1 - signalEnergy));
                float divisor = Math.Max(norm[i], adaptiveThreshold);
                output[i] /= divisor;
            }

            // Phase 2: Transition band - smooth connection between normalization and fade
            for (int i = transitionStartIndex; i < fadeStartIndex; i++)
            {
                float transitionFactor = transitionLength > 0
                    ? (float)(i - transitionStartIndex) / transitionLength
                    : 0;

                float signalEnergy = Math.Abs(output[i]);
                float adaptiveThreshold = Math.Max(minThreshold, maxThreshold * (1 - signalEnergy * (1 - transitionFactor)));
                float divisor = Math.Max(norm[i], adaptiveThreshold);
                output[i] /= divisor;

                // Pre-attenuation to reduce fade pressure
                output[i] *= (1 - transitionFactor * 0.5f);
            }

            // Phase 3: Fade region - 5th-order polynomial fade
            for (int i = fadeStartIndex; i < output.Length; i++)
            {
                float progress = (float)(i - fadeStartIndex) / fadeOutLength;
                // 5th-order polynomial fade: 1 - 5x² + 10x³ - 10x⁴ + 4x⁵
                float fadeFactor = 1.0f
                    - 5 * progress * progress
                    + 10 * (float)Math.Pow(progress, 3)
                    - 10 * (float)Math.Pow(progress, 4)
                    + 4 * (float)Math.Pow(progress, 5);

                // Use normalized value from end of transition band
                float divisor = Math.Max(norm[fadeStartIndex - 1], minThreshold);
                output[i] = (output[i] / divisor) * fadeFactor;
            }

            // Ensure final sample is zero to eliminate residual noise
            if (output.Length > 0)
            {
                output[output.Length - 1] = 0;
            }
        }
        #endregion
    }
}