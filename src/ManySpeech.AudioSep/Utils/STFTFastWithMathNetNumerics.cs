using System;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using Complex = System.Numerics.Complex;

namespace ManySpeech.AudioSep.Utils
{
    public class STFTFastWithMathNetNumerics
    {
        public static Complex[,,] ComputeSTFT(float[] input, int n_fft, int? hop_length = null,
                                             int? win_length = null, float[] window = null,
                                             bool center = true, string pad_mode = "reflect",
                                             bool normalized = false, bool? onesided = null,
                                             bool return_complex = true)
        {
            // 参数验证与设置
            hop_length = hop_length ?? n_fft / 4;
            win_length = win_length ?? n_fft;
            onesided = onesided ?? true;

            if (input == null || input.Length == 0)
                throw new ArgumentException("Input cannot be null or empty");
            if (n_fft <= 0 || hop_length <= 0 || win_length <= 0 || win_length > n_fft)
                throw new ArgumentException("Invalid FFT parameters");

            // 创建窗函数（不进行COLA归一化，与PyTorch一致）
            window = window ?? CreateHannWindow(win_length.Value);
            if (window.Length < n_fft)
            {
                window = PadWindowToSize(window, n_fft);
            }

            // Normalize window for COLA (Constant Overlap-Add) condition
            if (normalized)
            {
                float windowSum = window.Sum(x => x * x);
                float normalizationFactor = (float)Math.Sqrt(n_fft * windowSum / hop_length.Value);
                for (int i = 0; i < window.Length; i++)
                {
                    window[i] /= normalizationFactor;
                }
            }

            // 输入填充
            if (center)
            {
                input = PadInput(input, n_fft, pad_mode);
            }

            // 计算帧数和频率点数
            int n_frames = (input.Length - n_fft) / hop_length.Value + 1;
            int n_freqs = onesided.Value ? (n_fft / 2 + 1) : n_fft;

            // 初始化输出数组
            var stft_result = new Complex[n_freqs, 1, n_frames];

            // 并行处理每一帧
            Parallel.For(0, n_frames, () => new Complex[n_fft],
            (t, loopState, frameComplex) =>
            {
                // 提取当前帧并应用窗函数
                int offset = t * hop_length.Value;
                for (int i = 0; i < n_fft; i++)
                {
                    float sample = (offset + i < input.Length) ? input[offset + i] : 0f;
                    frameComplex[i] = new Complex(sample * window[i], 0);
                }

                // 执行FFT
                Fourier.Forward(frameComplex, FourierOptions.NoScaling);

                // 应用归一化（如果需要）
                if (normalized)
                {
                    double scale = 1.0 / Math.Sqrt(n_fft);
                    for (int i = 0; i < n_fft; i++)
                    {
                        frameComplex[i] *= scale;
                    }
                }

                // 存储结果
                if (onesided.Value)
                {
                    for (int f = 0; f < n_freqs; f++)
                    {
                        stft_result[f, 0, t] = frameComplex[f];
                    }
                }
                else
                {
                    for (int f = 0; f < n_fft; f++)
                    {
                        stft_result[f, 0, t] = frameComplex[f];
                    }
                }

                return frameComplex;
            },
            _ => { });

            return stft_result;
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

        // 窗口填充到指定大小
        private static float[] PadWindowToSize(float[] window, int targetLength)
        {
            if (window.Length == targetLength) return window;

            float[] padded = new float[targetLength];
            int padLeft = (targetLength - window.Length) / 2;
            Array.Copy(window, 0, padded, padLeft, window.Length);
            return padded;
        }

        // 输入填充（与torch.stft的pad_mode一致）
        private static float[] PadInput(float[] input, int n_fft, string mode)
        {
            int padWidth = n_fft / 2;
            float[] padded = new float[input.Length + 2 * padWidth];

            switch (mode)
            {
                case "reflect":
                    for (int i = 0; i < padWidth; i++)
                    {
                        padded[i] = input[padWidth - i - 1];
                        padded[padded.Length - 1 - i] = input[input.Length - 1 - (padWidth - i - 1)];
                    }
                    break;
                case "constant":
                    // 默认已初始化为0
                    break;
                case "replicate":
                    float firstVal = input[0];
                    float lastVal = input[input.Length - 1];
                    for (int i = 0; i < padWidth; i++)
                    {
                        padded[i] = firstVal;
                        padded[padded.Length - 1 - i] = lastVal;
                    }
                    break;
                case "circular":
                    for (int i = 0; i < padWidth; i++)
                    {
                        padded[i] = input[input.Length - padWidth + i];
                        padded[padded.Length - 1 - i] = input[padWidth - i - 1];
                    }
                    break;
                default:
                    throw new ArgumentException($"Unsupported pad mode: {mode}");
            }

            Array.Copy(input, 0, padded, padWidth, input.Length);
            return padded;
        }
    }
}
