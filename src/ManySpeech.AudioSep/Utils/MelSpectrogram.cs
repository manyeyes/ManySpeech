using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;

namespace ManySpeech.AudioSep.Utils
{
    public class MelSpectrogram
    {
        private static Dictionary<string, float[,]> melBasis = new Dictionary<string, float[,]>();
        private static Dictionary<string, float[]> hannWindow = new Dictionary<string, float[]>();

        public float[,] ComputeMelSpectrogram(float[] y, int nFft, int numMels, int samplingRate,
                                      int hopSize, int winSize, float fmin, float fmax,
                                      bool center = false)
        {
            // Check input range
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            foreach (var sample in y)
            {
                if (sample < minVal) minVal = sample;
                if (sample > maxVal) maxVal = sample;
            }

            if (minVal < -1.0f)
                Console.WriteLine($"min value is {minVal}");
            if (maxVal > 1.0f)
                Console.WriteLine($"max value is {maxVal}");

            // Generate or get Mel filterbank
            string melKey = $"{fmax}";
            if (!melBasis.ContainsKey(melKey))
            {
                var mel = MelFilterBank.Mel(samplingRate, nFft, numMels, fmin, fmax);
                melBasis[melKey] = mel;
            }

            // Generate or get Hann window
            string windowKey = $"{winSize}";
            if (!hannWindow.ContainsKey(windowKey))
            {
                hannWindow[windowKey] = HannWindow(winSize,true);
            }

            // Pad audio
            int padding = (nFft - hopSize) / 2;
            float[] paddedY = PadSignal(y, padding, padding);

            // Compute STFT using the provided method
            var stft = STFTFastWithMathNetNumerics.ComputeSTFT(
                input: paddedY,
                n_fft: nFft,
                hop_length: hopSize,
                win_length: winSize,
                window: hannWindow[windowKey],
                center: center,
                pad_mode: "reflect",
                normalized: false,
                onesided: true//,
                //return_complex: false
            );
            float[,,] spectrum = ConvertComplexToSTFTFormat(stft);

            // Compute magnitude spectrum
            //var magSpec = ComputeMagnitude(spectrum);
            var magSpec = CalculateSpec(spectrum);

            // Apply Mel filterbank
            var melSpec = ApplyMelFilterbank(melBasis[melKey], magSpec);
            //var melSpec = CalculateMelSpectrogramParallel(melBasis[melKey], magSpec);

            // Spectral normalization
            var normalizedSpec = SpectralNormalize(melSpec);

            return normalizedSpec;
        }

        /// <summary>
        /// Converts a Complex[961, 1, 1808] array to float[961, 1808, 2] STFT format
        /// </summary>
        /// <param name="complexSpectrogram">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [freq_bins, time_frames, 2]</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexArray)
        {
            int n_freq = complexArray.GetLength(0);  // 961 (频率bin)
            int n_channels = complexArray.GetLength(1); // 1 (单通道)
            int n_frames = complexArray.GetLength(2); // 1808 (时间帧)

            // 目标形状: [n_freq, n_frames, 2] (实部+虚部)
            float[,,] floatArray = new float[n_freq, n_frames, 2];

            for (int f = 0; f < n_freq; f++)
            {
                for (int t = 0; t < n_frames; t++)
                {
                    // 提取实部和虚部
                    floatArray[f, t, 0] = (float)complexArray[f, 0, t].Real; // 实部
                    floatArray[f, t, 1] = (float)complexArray[f, 0, t].Imaginary; // 虚部
                }
            }
            return floatArray;
        }

        // 转换为[batch, freq, time, 2]结构
        float[,,,] ToTorchFormat(Complex[,,] stft)
        {
            int n_freqs = stft.GetLength(0);
            int batch = stft.GetLength(1);
            int n_frames = stft.GetLength(2);
            float[,,,] result = new float[batch, n_freqs, n_frames, 2];

            for (int b = 0; b < batch; b++)
            {
                for (int f = 0; f < n_freqs; f++)
                {
                    for (int t = 0; t < n_frames; t++)
                    {
                        result[b, f, t, 0] = (float)stft[f, b, t].Real;
                        result[b, f, t, 1] = (float)stft[f, b, t].Imaginary;
                    }
                }
            }

            return result;
        }

        public static float[,] CalculateSpec(float[,,] input)
        {
            int dim0 = input.GetLength(0); // 513
            int dim1 = input.GetLength(1); // 2718
            int dim2 = input.GetLength(2); // 2 (should be 2 for this operation)

            float[,] result = new float[dim0, dim1];
            const double epsilon = 1e-9;

            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    // Sum of squares along the last dimension
                    double sumOfSquares = 0;
                    for (int k = 0; k < dim2; k++)
                    {
                        sumOfSquares += Math.Pow(input[i, j, k], 2);
                    }

                    // Square root of sum plus epsilon
                    result[i, j] = (float)Math.Sqrt(sumOfSquares + epsilon);
                }
            }

            return result;
        }

        public static float[,] CalculateMelSpectrogramParallel(float[,] melBasis, float[,] spec)
        {
            int melBands = melBasis.GetLength(0); // 80
            int fftSize = melBasis.GetLength(1);  // 513
            int frames = spec.GetLength(1);       // 2718

            if (fftSize != spec.GetLength(0))
            {
                throw new ArgumentException("Matrix dimensions don't match for multiplication");
            }

            float[,] result = new float[melBands, frames];

            Parallel.For(0, melBands, i =>
            {
                for (int j = 0; j < frames; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < fftSize; k++)
                    {
                        sum += melBasis[i, k] * spec[k, j];
                    }
                    result[i, j] = (float)sum;
                }
            });

            return result;
        }

        private float[,] LibrosaMelFilter(int sr, int nFft, int nMel, float fmin, float fmax)
        {
            // Implementation of Mel filterbank similar to librosa
            float[] freqs = new float[nMel + 2];
            float melMin = HzToMel(fmin);
            float melMax = HzToMel(fmax);

            for (int i = 0; i < nMel + 2; i++)
            {
                freqs[i] = MelToHz(melMin + i * (melMax - melMin) / (nMel + 1));
            }

            float[] fftFreqs = new float[nFft / 2 + 1];
            for (int i = 0; i < fftFreqs.Length; i++)
            {
                fftFreqs[i] = i * sr / (float)nFft;
            }

            float[,] filter = new float[nMel, nFft / 2 + 1];

            for (int i = 0; i < nMel; i++)
            {
                for (int j = 0; j < fftFreqs.Length; j++)
                {
                    float freq = fftFreqs[j];
                    if (freq >= freqs[i] && freq <= freqs[i + 1])
                    {
                        filter[i, j] = (freq - freqs[i]) / (freqs[i + 1] - freqs[i]);
                    }
                    else if (freq >= freqs[i + 1] && freq <= freqs[i + 2])
                    {
                        filter[i, j] = (freqs[i + 2] - freq) / (freqs[i + 2] - freqs[i + 1]);
                    }
                }
            }

            // Normalize filters
            for (int i = 0; i < nMel; i++)
            {
                float sum = 0;
                for (int j = 0; j < fftFreqs.Length; j++)
                {
                    sum += filter[i, j];
                }
                if (sum > 0)
                {
                    for (int j = 0; j < fftFreqs.Length; j++)
                    {
                        filter[i, j] /= sum;
                    }
                }
            }

            return filter;
        }

        private float HzToMel(float hz)
        {
            return 2595f * (float)Math.Log10(1 + hz / 700f);
        }

        private float MelToHz(float mel)
        {
            return 700f * ((float)Math.Pow(10, mel / 2595f) - 1);
        }

        private float[] HannWindow(int length)
        {
            float[] window = new float[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5f * (1 - (float)Math.Cos(2 * Math.PI * i / (length - 1)));
            }
            return window;
        }

        /// <summary>
        /// Hanning窗函数，与torch.hann_window功能一致
        /// </summary>
        /// <param name="windowLength">窗函数长度</param>
        /// <param name="periodic">是否为周期性信号优化</param>
        /// <returns>Hanning窗函数数组</returns>
        public static float[] HannWindow(int windowLength, bool periodic = true)
        {
            if (windowLength < 0)
                throw new ArgumentException("Window length must be non-negative");

            int length = periodic ? windowLength + 1 : windowLength;
            float[] window = new float[windowLength];

            for (int i = 0; i < windowLength; i++)
            {
                double angle = 2.0 * Math.PI * i / (length - 1);
                window[i] = 0.5f * (1.0f - (float)Math.Cos(angle));
            }

            return window;
        }

        private float[] PadSignal(float[] signal, int leftPad, int rightPad)
        {
            float[] padded = new float[signal.Length + leftPad + rightPad];

            // Reflect padding for left side
            for (int i = 0; i < leftPad; i++)
            {
                padded[i] = signal[leftPad - i];
            }

            // Copy original signal
            Array.Copy(signal, 0, padded, leftPad, signal.Length);

            // Reflect padding for right side
            for (int i = 0; i < rightPad; i++)
            {
                padded[leftPad + signal.Length + i] = signal[signal.Length - 2 - i];
            }

            return padded;
        }
        private float[,] ComputeMagnitude(float[,,] stft)
        {
            // 原始维度顺序: [batch, freq, time] -> 调整为: [time, freq, ?]
            //float[,,] permutedMask = AudioProcessing.PermuteDimensions(predMask, 2, 1, 0);
            //float[,,] stftF = AudioProcessing.PermuteDimensions(stft, 1, 0, 2);
            // Assuming stft dimensions are [1, freq_bins, time_frames]
            int freqBins = stft.GetLength(0);
            int timeFrames = stft.GetLength(1);
            float[,] magnitude = new float[freqBins, timeFrames];

            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    magnitude[f, t] = (float)Complex.Abs(stft[f, t,0]);
                }
            }

            return magnitude;
        }
        private float[,] ComputeMagnitude(Complex[,,] stft)
        {
            // 原始维度顺序: [batch, freq, time] -> 调整为: [time, freq, ?]
            //float[,,] permutedMask = AudioProcessing.PermuteDimensions(predMask, 2, 1, 0);
            float[,,] stftF = AudioProcessing.PermuteDimensions(stft, 1, 0, 2);
            // Assuming stft dimensions are [1, freq_bins, time_frames]
            int freqBins = stftF.GetLength(1);
            int timeFrames = stftF.GetLength(2);
            float[,] magnitude = new float[freqBins, timeFrames];

            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    magnitude[f, t] = (float)Complex.Abs(stftF[0, f, t]);
                }
            }

            return magnitude;
        }

        private float[,] ApplyMelFilterbank(float[,] melFilter, float[,] spectrogram)
        {
            int melBins = melFilter.GetLength(0);
            int timeFrames = spectrogram.GetLength(1);
            float[,] melSpec = new float[melBins, timeFrames];

            for (int m = 0; m < melBins; m++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    float sum = 0;
                    for (int f = 0; f < spectrogram.GetLength(0); f++)
                    {
                        sum += melFilter[m, f] * spectrogram[f, t];
                    }
                    melSpec[m, t] = sum;
                }
            }

            return melSpec;
        }

        private float[,] SpectralNormalize(float[,] magnitudes)
        {
            return DynamicRangeCompression(magnitudes);
        }

        private float[,] DynamicRangeCompression(float[,] x, float C = 1f, float clipVal = 1e-5f)
        {
            int rows = x.GetLength(0);
            int cols = x.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float val = Math.Max(x[i, j], clipVal);
                    result[i, j] = (float)Math.Log(val * C);
                }
            }

            return result;
        }
    }
}