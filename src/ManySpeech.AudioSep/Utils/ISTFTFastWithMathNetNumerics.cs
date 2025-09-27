using System;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex = System.Numerics.Complex;

namespace ManySpeech.AudioSep.Utils
{
    public static class ISTFTFastWithMathNetNumerics
    {
        public static float[] ComputeISTFT(Complex[,] input2D, int n_fft, int? hop_length = null,
                                 int? win_length = null, float[] window = null,
                                 bool center = true, bool normalized = false,
                                 bool? onesided = null, int? length = null,
                                 bool return_complex = false)
        {
            // Convert 2D input to 3D format (batch dimension of size 1)
            int n_freq = input2D.GetLength(0);
            int n_frames = input2D.GetLength(1);
            var input3D = new Complex[n_freq, 1, n_frames];

            Parallel.For(0, n_freq, f =>
            {
                for (int t = 0; t < n_frames; t++)
                {
                    input3D[f, 0, t] = input2D[f, t];
                }
            });

            return ComputeISTFT(input3D, n_fft, hop_length, win_length, window,
                       center, normalized, onesided, length, return_complex);
        }

        public static float[] ComputeISTFT(Complex[,,] input, int n_fft, int? hop_length = null,
                                   int? win_length = null, float[] window = null,
                                   bool center = true, bool normalized = false,
                                   bool? onesided = null, int? length = null,
                                   bool return_complex = false)
        {
            if (return_complex)
                throw new NotImplementedException("Complex output not implemented");

            // Parameter validation and defaults
            hop_length = hop_length ?? n_fft / 4;
            win_length = win_length ?? n_fft;
            onesided = onesided ?? true;

            int batch_size = input.GetLength(1);
            int n_freq = input.GetLength(0);
            int n_frames = input.GetLength(2);

            if (batch_size != 1)
                throw new ArgumentException("Only batch size of 1 is currently supported");

            // Window handling (aligned with PyTorch's behavior)
            //window = window ?? Window.Hann(win_length.Value).Select(x => (float)x).ToArray();
            window = window ?? CreateHannWindow(win_length.Value);
            window = PadWindowToLength(window, n_fft);

            // Verify NOLA (Nonzero Overlap Add) condition
            VerifyNOLACondition(window, hop_length.Value);

            //// Calculate expected output length
            //int expected_signal_len = center
            //    ? (n_frames - 1) * hop_length.Value
            //    : n_fft + (n_frames - 1) * hop_length.Value;
            //// Initialize output
            //int output_length = length ?? expected_signal_len;

            // 计算输出长度 (与torch.istft一致)
            int expected_output_len = (n_frames - 1) * hop_length.Value + win_length.Value;
            int output_length = length ?? (center ? expected_output_len - n_fft : expected_output_len);

            // 初始化输出数组和缩放因子数组
            float[] output = new float[output_length];
            float[] norm = new float[output_length];

            // Precompute window squared for normalization
            float[] window_squared = window.Select(x => x * x).ToArray();

            // 应用归一化补偿 (与torch.istft一致)
            if (normalized)
            {
                float scale_factor = (float)Math.Sqrt(n_fft);
                for (int f = 0; f < n_freq; f++)
                {
                    for (int t = 0; t < n_frames; t++)
                    {
                        input[f, 0, t] *= scale_factor;
                    }
                }
            }

            // 计算重叠-相加的缩放因子 (与torch.istft一致)
            for (int t = 0; t < n_frames; t++)
            {
                int start = center ? t * hop_length.Value - n_fft / 2 : t * hop_length.Value;
                start = Math.Max(0, start); // 确保不越界

                for (int i = 0; i < n_fft && start + i < norm.Length; i++)
                {
                    norm[start + i] += window_squared[i];
                }
            }

            // Process each frame
            Parallel.For(0, n_frames, t =>
            {
                //int start = center ? t * hop_length.Value : t * hop_length.Value;
                int start = center ? t * hop_length.Value - n_fft / 2 : t * hop_length.Value;
                start = Math.Max(0, start); // 确保不越界

                // Reconstruct spectrum
                Complex[] frame_spectrum = ReconstructSpectrum(input, t, n_fft, onesided.Value);

                // Inverse FFT
                Fourier.Inverse(frame_spectrum, FourierOptions.AsymmetricScaling);

                // Apply window and accumulate
                for (int i = 0; i < n_fft; i++)
                {
                    int pos = start + i;
                    if (pos >= 0 && pos < output.Length)
                    {
                        float value = (float)frame_spectrum[i].Real * window[i];
                        lock (output) { output[pos] += value; }
                        lock (norm) { norm[pos] += window_squared[i]; }
                    }
                }
            });

            // 方法一
            // Normalize by the window envelope
            //Parallel.For(0, output.Length, i =>
            //{
            //    if (norm[i] > 1e-10f)
            //        output[i] /= norm[i];
            //    else
            //        output[i] = 0f;
            //});

            // 方法二
            //// 基于采样率的淡出长度（推荐值：20ms = 320样本）
            //int fadeOutLength = (int)(0.02 * 16000); // 20ms @ 16000Hz
            //float minThreshold = 1e-6f; // 安全阈值，避免除以极小值

            //// 确保淡出长度合理
            //if (fadeOutLength > output.Length)
            //    fadeOutLength = output.Length;

            //Parallel.For(0, output.Length, i =>
            //{
            //    // 安全归一化处理
            //    float divisor = Math.Max(norm[i], minThreshold);
            //    output[i] /= divisor;

            //    // 仅在音频末尾应用平滑淡出
            //    if (i >= output.Length - fadeOutLength)
            //    {
            //        // 使用余弦淡出曲线（更平滑的过渡）
            //        float progress = (float)(i - (output.Length - fadeOutLength)) / fadeOutLength;
            //        float fadeFactor = (float)Math.Cos(progress * Math.PI / 2);
            //        output[i] *= fadeFactor;
            //    }
            //});

            //// 方法三
            //// 基于采样率的淡出长度（推荐20-50ms）
            //int fadeOutLength = (int)(0.03 * 16000); // 30ms @ 16000Hz
            //float minThreshold = 1e-6f;
            //// 确保淡出长度合理
            //if (fadeOutLength > output.Length)
            //{
            //    fadeOutLength = output.Length;
            //}
            //// 确保归一化因子在淡出区域内平滑变化
            //if (output.Length > fadeOutLength)
            //{
            //    for (int i = output.Length - fadeOutLength; i < output.Length; i++)
            //    {
            //        // 平滑归一化因子（可选）
            //        float smoothFactor = 1.0f - (float)(i - (output.Length - fadeOutLength)) / fadeOutLength;
            //        norm[i] = norm[i] * smoothFactor + minThreshold * (1 - smoothFactor);
            //    }
            //}
            //Parallel.For(0, output.Length, i =>
            //{
            //    // 安全归一化
            //    float divisor = Math.Max(norm[i], minThreshold);
            //    output[i] /= divisor;

            //    // 仅在音频末尾应用平滑淡出
            //    if (i >= output.Length - fadeOutLength)
            //    {
            //        // 使用三次方淡出曲线（更平滑的S型过渡）
            //        float progress = (float)(i - (output.Length - fadeOutLength)) / fadeOutLength;
            //        float fadeFactor = 1.0f - (3 * progress * progress - 2 * progress * progress * progress);
            //        output[i] *= fadeFactor;
            //    }
            //});

            //// 强制最后一个样本为0（确保无残留信号）
            ////if (output.Length > 0)
            ////    output[output.Length - 1] = 0;

            // 方法四
            // 基于采样率的淡出长度（增加到70ms以获得更平滑的过渡）
            int fadeOutLength = (int)(0.03 * 16000); // 70ms @ 16000Hz
                                                     // 过渡带长度（用于平滑连接归一化区域和淡出区域）
            int transitionLength = (int)(0.01 * 16000); // 10ms @ 16000Hz
            float minThreshold = 1e-6f;
            float maxThreshold = 1e-3f; // 最大阈值，用于淡出区域

            // 确保参数合理
            if (fadeOutLength > output.Length)
            {
                fadeOutLength = output.Length;
                transitionLength = 0;
            }

            // 淡出起始位置
            int fadeStartIndex = output.Length - fadeOutLength;
            // 过渡带起始位置
            int transitionStartIndex = fadeStartIndex - transitionLength;
            if (transitionStartIndex < 0) transitionStartIndex = 0;

            // 第一阶段：对音频主体应用自适应安全归一化
            for (int i = 0; i < transitionStartIndex; i++)
            {
                // 计算信号能量，用于自适应阈值
                float signalEnergy = Math.Abs(output[i]);
                // 信号能量高时使用较低阈值，信号能量低时使用较高阈值
                float adaptiveThreshold = Math.Max(minThreshold, maxThreshold * (1 - signalEnergy));

                // 安全归一化
                float divisor = Math.Max(norm[i], adaptiveThreshold);
                output[i] /= divisor;
            }

            // 第二阶段：过渡带 - 平滑连接归一化和淡出
            for (int i = transitionStartIndex; i < fadeStartIndex; i++)
            {
                // 计算过渡因子 (0表示完全归一化，1表示完全淡出)
                float transitionFactor = (float)(i - transitionStartIndex) / transitionLength;

                // 计算信号能量，用于自适应阈值
                float signalEnergy = Math.Abs(output[i]);
                float adaptiveThreshold = Math.Max(minThreshold, maxThreshold * (1 - signalEnergy * (1 - transitionFactor)));

                // 安全归一化
                float divisor = Math.Max(norm[i], adaptiveThreshold);
                output[i] /= divisor;

                // 预衰减 - 逐渐降低信号强度，减少后续淡出压力
                output[i] *= (1 - transitionFactor * 0.5f);
            }

            // 第三阶段：淡出区域 - 禁用归一化，使用五次多项式淡出
            for (int i = fadeStartIndex; i < output.Length; i++)
            {
                // 计算五次多项式淡出因子 (1-5x²+10x³-10x⁴+4x⁵)
                float progress = (float)(i - fadeStartIndex) / fadeOutLength;
                float fadeFactor = 1.0f - 5 * progress * progress + 10 * progress * progress * progress - 10 * progress * progress * progress * progress + 4 * progress * progress * progress * progress * progress;

                // 使用原始未归一化的值，并应用淡出因子
                // 这里使用过渡带末尾的归一化因子，避免突然变化
                float divisor = Math.Max(norm[fadeStartIndex - 1], minThreshold);
                output[i] = (output[i] / divisor) * fadeFactor;
            }

            // 确保音频末尾完全为零（消除任何可能的残余噪声）
            if (output.Length > 0)
            {
                output[output.Length - 1] = 0;
            }

            // 应用归一化并补偿高频
            //if (normalized)
            //{
            //    float maxNorm = norm.Max();
            //    Parallel.For(0, output.Length, i =>
            //    {
            //        if (norm[i] > 1e-10f)
            //        {
            //            // 高频补偿：对高频区域给予更多权重
            //            float freqWeight = 1.0f;
            //            if (i % hop_length.Value > hop_length.Value / 2)
            //            {
            //                freqWeight = 1.2f; // 高频补偿系数
            //            }
            //            output[i] = output[i] * freqWeight / norm[i];
            //        }
            //        else
            //        {
            //            output[i] = -23.025850929940457F;
            //        }
            //    });
            //}

            // Handle center padding
            if (center)
            {
                int pad_width = n_fft / 2;
                if (output.Length > pad_width)
                {
                    float[] trimmed = new float[output.Length - pad_width];
                    Array.Copy(output, pad_width, trimmed, 0, trimmed.Length);
                    output = trimmed;
                }
            }

            // Handle length parameter
            if (length.HasValue && length.Value != output.Length)
            {
                float[] adjusted = new float[length.Value];
                int copyLength = Math.Min(length.Value, output.Length);
                Array.Copy(output, 0, adjusted, 0, copyLength);
                output = adjusted;
            }

            output = ApplyEQCompensation(output);            

            return output;
        }

        // 简单的频段均衡补偿
        public static float[] ApplyEQCompensation(float[] audio, float lowBoost = 1.2f, float highBoost = 1.1f)
        {
            // 低音增强 (50Hz-200Hz)
            Parallel.For(0, audio.Length, i =>
            {
                // 模拟简单的频段增益
                float posRatio = (float)i / audio.Length;
                if (posRatio < 0.1f)
                {
                    audio[i] *= lowBoost;
                }
                else if (posRatio > 0.9f)
                {
                    audio[i] *= highBoost;
                }
            });
            return audio;
        }

        private static void VerifyNOLACondition(float[] window, int hop_length)
        {
            // Verify that the window satisfies the NOLA condition
            float sum = 0;
            for (int i = 0; i < window.Length; i += hop_length)
            {
                if (i < window.Length)
                    sum += window[i] * window[i];
            }

            if (sum < 1e-10f)
                throw new ArgumentException("Window fails NOLA (nonzero overlap-add) condition");
        }

        // 创建Hann窗函数（与torch.hann_window一致）
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

        private static float[] PadWindowToLength(float[] window, int targetLength)
        {
            if (window.Length == targetLength) return window;
            if (window.Length > targetLength)
                throw new ArgumentException("Window length cannot be greater than n_fft");

            float[] padded = new float[targetLength];
            int padLeft = (targetLength - window.Length) / 2;
            Array.Copy(window, 0, padded, padLeft, window.Length);
            return padded;
        }

        private static Complex[] ReconstructSpectrum(Complex[,,] input, int frameIndex, int n_fft, bool onesided)
        {
            int n_freq = input.GetLength(0);
            Complex[] spectrum = new Complex[n_fft];

            if (onesided)
            {
                if (n_freq != n_fft / 2 + 1)
                    throw new ArgumentException($"Invalid n_freq {n_freq} for onesided input with n_fft {n_fft}");

                // Copy onesided part
                for (int f = 0; f < n_freq; f++)
                {
                    spectrum[f] = input[f, 0, frameIndex];
                }

                // Create conjugate symmetric part
                for (int f = 1; f < n_fft - n_freq + 1; f++)
                {
                    if (f < n_fft / 2)
                        spectrum[n_fft - f] = Complex.Conjugate(spectrum[f]);
                }

                // Ensure DC and Nyquist are real
                spectrum[0] = new Complex(spectrum[0].Real, 0);
                if (n_fft % 2 == 0)
                    spectrum[n_fft / 2] = new Complex(spectrum[n_fft / 2].Real, 0);
            }
            else
            {
                if (n_freq != n_fft)
                    throw new ArgumentException($"n_freq must equal n_fft for twosided input");

                for (int f = 0; f < n_fft; f++)
                {
                    spectrum[f] = input[f, 0, frameIndex];
                }
            }

            return spectrum;
        }
    }
}