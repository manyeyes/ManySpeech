using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;

namespace ManySpeech.AudioSep.Utils
{
    public class STFTArgs
    {
        public string win_type { get; set; }
        public int win_len { get; set; }
        public int win_inc { get; set; }
        public int fft_len { get; set; }
    }

    public class MelArgs
    {
        public int n_fft { get; set; } = 1024;
        public int num_mels { get; set; } = 80;
        public int hop_size { get; set; } = 256;
        public int win_size { get; set; } = 1024;
        public int sampling_rate { get; set; } = 48000;
        public int fmin { get; set; } = 0;
        public int fmax { get; set; } = 8000;
        public bool center { get; set; } = false;
    }
    public class AudioProcessing
    {
        public static float[,] MelSpec(float[] y, MelArgs args)
        {
            float[,] melSpec = new Utils.MelSpectrogram().ComputeMelSpectrogram(y: y, nFft: args.n_fft, numMels: args.num_mels, samplingRate: args.sampling_rate, hopSize: args.hop_size, winSize: args.win_size, fmin: args.fmin, fmax: args.fmax, center: args.center);
            return melSpec;

        }

        // 计算短时傅里叶变换 (STFT)
        public static Complex[,,] Stft(float[] x, STFTArgs args, bool center = false, bool periodic = false, bool? onesided = false, bool normalized = false, string pad_mode = "reflect")
        {
            string winType = args.win_type;
            int winLen = args.win_len;
            int winInc = args.win_inc;
            int fftLen = args.fft_len;

            // 选择窗函数类型并创建窗函数
            float[] window = CreateWindow(winType, winLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In STFT, {winType} is not supported!");
                return null;
            }

            // 初始化STFT结果矩阵
            Complex[,,] stftComplex = STFTFastWithMathNetNumerics.ComputeSTFT(
                input: x, n_fft:
                fftLen,
                hop_length: winInc,
                win_length: winLen,
                center: center,
                window: window,
                normalized: normalized,
                pad_mode: pad_mode,
                onesided: true);
            return stftComplex;
        }

        /// <summary>
        /// Converts back from float[1, 642, 723] STFT format to Complex[321, 2, 723]
        /// </summary>
        /// <param name="stftFormat">STFT format array with shape [1, 2*freq_bins, time_frames]</param>
        /// <returns>Complex spectrogram with shape [freq_bins, 1, time_frames]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex2(float[,,] stftFormat)
        {
            //int totalChannels = stftFormat.GetLength(1);
            int freqBins = stftFormat.GetLength(1) / 2;
            int timeFrames = stftFormat.GetLength(2);
            int channels = 2;

            Complex[,,] complexSpectrogram = new Complex[freqBins, channels, timeFrames];

            //var complexData = new Complex[freqBins, channels, timeFrames];

            for (int f = 0; f < freqBins; f++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // For channel 0: use first 321 elements (real parts)
                        // For channel 1: use next 321 elements (imaginary parts)
                        int sourceIndex = c == 0 ? f : f + freqBins;
                        complexSpectrogram[f, c, t] = new Complex(
                            stftFormat[0, sourceIndex, t],
                            0); // Initialize with 0 imaginary
                    }
                }
            }

            return complexSpectrogram;
        }

        // 计算逆短时傅里叶变换 (ISTFT)
        public static float[] Istft(Complex[,] x, STFTArgs args, int? slen = null, bool center = false,
                                   bool normalized = false, bool periodic = false, bool? onesided = null,
                                   bool return_complex = false, float[]? window = null)
        {
            string winType = args.win_type;
            int winLen = args.win_len;
            int winInc = args.win_inc;
            int fftLen = args.fft_len;

            // 选择窗函数类型并创建窗函数
            window = window ?? CreateWindow(winType, winLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In ISTFT, {winType} is not supported!");
                return null;
            }

            // 初始化输出信号和归一化缓冲区
            float[] output = ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input2D: x,
                n_fft: fftLen,
                hop_length: winInc,
                win_length: winLen,
                window: window, // Uses Hann window
                center: true,
                normalized: normalized,
                onesided: true,
                length: slen // Reconstruct original length
            );

            return output;
        }

        public static float[] Istft(Complex[,,] x, STFTArgs args, int? slen = null, bool center = false,
                                   bool normalized = false, bool periodic = false, bool? onesided = null,
                                   bool return_complex = false, float[]? window = null)
        {
            string winType = args.win_type;
            int winLen = args.win_len;
            int winInc = args.win_inc;
            int fftLen = args.fft_len;

            // 选择窗函数类型并创建窗函数
            window = window ?? CreateWindow(winType, winLen, periodic);
            if (window == null)
            {
                Console.WriteLine($"In ISTFT, {winType} is not supported!");
                return null;
            }

            // 初始化输出信号和归一化缓冲区
            float[] output = ISTFTFastWithMathNetNumerics.ComputeISTFT(
                input: x,
                n_fft: fftLen,
                hop_length: winInc,
                win_length: winLen,
                window: window, // Uses Hann window
                center: true,
                normalized: normalized,
                onesided: true,
                length: slen // Reconstruct original length
            );

            return output;
        }     

        // 创建窗函数
        private static float[] CreateWindow(string winType, int winLen, bool periodic)
        {
            float[] window = new float[winLen];

            if (winType == "hamming")
            {
                for (int i = 0; i < winLen; i++)
                {
                    window[i] = 0.54f - 0.46f * (float)Math.Cos(2 * Math.PI * i / (winLen - 1));
                }
            }
            else if (winType == "hanning")
            {
                for (int i = 0; i < winLen; i++)
                {
                    window[i] = 0.5f * (1 - (float)Math.Cos(2 * Math.PI * i / (winLen - 1)));
                }
            }
            else
            {
                return null;
            }

            return window;
        }
        /// <summary>
        /// 将一维数组重复为指定形状的三维数组 [1, windowLength, repeatCount]
        /// </summary>
        /// <param name="window">原始一维数组，长度为 windowLength</param>
        /// <param name="repeatCount">第三维重复次数</param>
        /// <returns>形状为 [1, windowLength, repeatCount] 的三维数组</returns>
        public static float[,,] RepeatTo3DArray(float[] window, int repeatCount)
        {
            if (window == null || window.Length == 0)
                throw new ArgumentException("window 数组不能为空");

            int windowLength = window.Length;
            // 创建目标三维数组 [1, windowLength, repeatCount]
            float[,,] result = new float[1, windowLength, repeatCount];

            // 填充数据：将window数组复制到第三维的每个位置
            for (int d = 0; d < repeatCount; d++)
            {
                for (int i = 0; i < windowLength; i++)
                {
                    result[0, i, d] = window[i];
                }
            }
            return result;
        }

        public static float[,] PermuteDimensions(float[,] tensor, int dim0, int dim1)
        {
            // 验证维度参数
            if (dim0 < 0 || dim0 > 1 || dim1 < 0 || dim1 > 1)
                throw new ArgumentException("维度参数必须是0或1");

            // 获取新维度的大小
            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);

            // 创建结果数组
            float[,] result = new float[size0, size1];

            // 遍历原始数组并根据新维度顺序填充结果
            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    // 根据指定的维度顺序映射索引
                    int newI = dim0 == 0 ? i : j;
                    int newJ = dim1 == 0 ? i : j;

                    result[newI, newJ] = tensor[i, j];
                }
            }

            return result;
        }

        // 辅助函数：调整张量维度顺序
        public static float[,,] PermuteDimensions(float[,,] tensor, int dim0, int dim1, int dim2)
        {
            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);
            int size2 = tensor.GetLength(dim2);

            float[,,] result = new float[size0, size1, size2];

            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    for (int k = 0; k < tensor.GetLength(2); k++)
                    {
                        int newI = i;
                        int newJ = j;
                        int newK = k;

                        if (dim0 == 0) newI = i;
                        else if (dim0 == 1) newI = j;
                        else if (dim0 == 2) newI = k;

                        if (dim1 == 0) newJ = i;
                        else if (dim1 == 1) newJ = j;
                        else if (dim1 == 2) newJ = k;

                        if (dim2 == 0) newK = i;
                        else if (dim2 == 1) newK = j;
                        else if (dim2 == 2) newK = k;

                        result[newI, newJ, newK] = tensor[i, j, k];
                    }
                }
            }

            return result;
        }
        public static float[,,] PermuteDimensions(Complex[,,] tensor, int dim0, int dim1, int dim2)
        {
            int size0 = tensor.GetLength(dim0);
            int size1 = tensor.GetLength(dim1);
            int size2 = tensor.GetLength(dim2);

            float[,,] result = new float[size0, size1, size2];

            for (int i = 0; i < tensor.GetLength(0); i++)
            {
                for (int j = 0; j < tensor.GetLength(1); j++)
                {
                    for (int k = 0; k < tensor.GetLength(2); k++)
                    {
                        int newI = i;
                        int newJ = j;
                        int newK = k;

                        if (dim0 == 0) newI = i;
                        else if (dim0 == 1) newI = j;
                        else if (dim0 == 2) newI = k;

                        if (dim1 == 0) newJ = i;
                        else if (dim1 == 1) newJ = j;
                        else if (dim1 == 2) newJ = k;

                        if (dim2 == 0) newK = i;
                        else if (dim2 == 1) newK = j;
                        else if (dim2 == 2) newK = k;

                        result[newI, newJ, newK] = (float)tensor[i, j, k].Real;
                    }
                }
            }

            return result;
        }

        // 辅助函数：将掩码应用到频谱上
        public static float[,,] ApplyMask(float[,,] spectrum, float[,,] mask)
        {
            int timeBins = spectrum.GetLength(0);
            int freqBins = spectrum.GetLength(1);

            // 验证掩码维度是否匹配
            if (mask.GetLength(0) != timeBins || mask.GetLength(1) != freqBins || mask.GetLength(2) != 1)
            {
                //throw new ArgumentException("掩码维度与频谱不匹配");
            }

            // 创建结果数组，保持与输入频谱相同的格式 [time, freq, 2]
            float[,,] result = new float[timeBins, freqBins, 2];

            for (int t = 0; t < timeBins; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    // 获取单通道掩码值（假设mask的第三维大小为1）
                    float maskValue = mask[t, f, 0];

                    // 应用掩码到频谱的实部和虚部
                    result[t, f, 0] = spectrum[t, f, 0] * maskValue;  // 实部
                    result[t, f, 1] = spectrum[t, f, 1] * maskValue;  // 虚部
                }
            }

            return result;
        }

        // 将float[,,]格式的复数频谱转换为Complex[,]格式
        public static Complex[,] ConvertToComplex(float[,,] spec)
        {
            int timeBins = spec.GetLength(0);
            int freqBins = spec.GetLength(1);

            // 创建复数数组
            Complex[,] complexSpec = new Complex[timeBins, freqBins];

            // 填充复数数组
            for (int t = 0; t < timeBins; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    // 从第三维获取实部和虚部
                    float real = spec[t, f, 0];
                    float imag = spec[t, f, 1];

                    // 创建复数
                    complexSpec[t, f] = new Complex(real, imag);
                }
            }

            return complexSpec;
        }

        public static float[,,] TensorTo3DArray(Tensor<float> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional.");

            int dim0 = tensor.Dimensions[0];
            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];

            float[,,] array3D = new float[dim0, dim1, dim2];

            // 计算每个维度的步长
            int stride0 = dim1 * dim2;
            int stride1 = dim2;
            int stride2 = 1;

            // 遍历所有元素
            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        // 计算线性索引
                        int linearIndex = i * stride0 + j * stride1 + k * stride2;
                        array3D[i, j, k] = tensor[i, j, k];
                    }
                }
            }

            return array3D;
        }

        public static float[,,] TensorTo3DArray(DenseTensor<float> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional.");

            int dim0 = tensor.Dimensions[0];
            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];

            float[,,] array3D = new float[dim0, dim1, dim2];

            // 获取张量的内存span
            var tensorSpan = tensor.Buffer.Span;

            // 使用线性索引快速填充数组
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

        // 计算 float[] 的 RMS 值
        public static float CalculateRms(float[] data)
        {
            double sumSquared = 0;
            foreach (var value in data)
            {
                sumSquared += value * value;
            }
            return (float)Math.Sqrt(sumSquared / data.Length);
        }

        /// <summary>
        /// 使用输入的 RMS 值对样本进行归一化
        /// Normalize the outputs back to the input magnitude for each speaker
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="rmsInput"></param>
        /// <returns></returns>
        public static float[] NormalizeSample(float[] sample, float? rmsInput = null)
        {
            // 如果未提供 rmsInput，则使用输入样本自身的 RMS
            float effectiveRmsInput = rmsInput ?? CalculateRms(sample);

            // 计算样本的 RMS
            float rmsOut = CalculateRms(sample);

            // 避免除零错误
            if (rmsOut < 1e-10f)
            {
                rmsOut = 1e-10f;
            }

            // 归一化处理
            float[] result = new float[sample.Length];
            for (int i = 0; i < sample.Length; i++)
            {
                result[i] = sample[i] / rmsOut * effectiveRmsInput;
            }

            return result;
        }
    }
}