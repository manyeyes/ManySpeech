using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.LinearAlgebra;
using Complex = System.Numerics.Complex; // 使用System.Numerics.Complex更精确

namespace ManySpeech.AudioSep.Utils
{

    public static class SignalProcessing
    {
        /// <summary>
        /// 设计N阶Butterworth滤波器，与SciPy的signal.butter完全一致
        /// </summary>
        public static (double[] b, double[] a) Butter(int N, double[] Wn, string btype = "lowpass", bool analog = false, double fs = double.NaN)
        {
            // 参数验证与处理
            btype = btype.ToLower();
            var wnArray = Wn.ToArray();
            double fsInternal = analog ? double.NaN : (double.IsNaN(fs) ? 2.0 : fs);

            if (!analog)
            {
                if (wnArray.Any(w => w <= 0 || w >= fsInternal / 2))
                    throw new ArgumentException($"数字滤波器的Wn必须在(0, {fsInternal / 2})范围内（fs={fsInternal}）");
                wnArray = wnArray.Select(w => 2 * w / fsInternal).ToArray(); // 归一化到[0,1]
            }

            // 设计模拟低通原型
            var (zAnalog, pAnalog, kAnalog) = Buttap(N);

            // 频率变换
            double[] warped = analog ? wnArray : Warp(wnArray, fsInternal);
            //double[] warped = new double[] { 0.8982300372685172 };// analog ? wnArray : Warp(wnArray, fsInternal);
            var (zTrans, pTrans, kTrans) = TransformFilter(zAnalog, pAnalog, kAnalog, btype, warped, analog);

            // 双线性变换（数字滤波器）
            var (zDigital, pDigital, kDigital) = analog
                ? (zTrans, pTrans, kTrans)
                : BilinearZpk(zTrans, pTrans, kTrans, fsInternal);

            // 增益归一化
            double gain = ComputeNormalizationGain(zDigital, pDigital, kDigital, btype);
            kDigital /= gain;

            // 转换为分子/分母多项式
            return ZpkToBa(zDigital, pDigital, kDigital);
        }

        /// <summary>
        /// Butterworth模拟低通原型滤波器的极点和零点
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) Buttap(int N)
        {
            var poles = new Complex[N];
            for (int i = 0; i < N; i++)
            {
                double angle = Math.PI * (2 * i + N + 1) / (2 * N);
                poles[i] = Complex.Exp(Complex.ImaginaryOne * angle);
            }
            return (Array.Empty<Complex>(), poles, 1.0);
        }

        /// <summary>
        /// 频率预畸变（数字滤波器设计中使用）
        /// </summary>
        private static double[] Warp(double[] wn, double fs)
        {
            //return wn.Select(w => 2 * fs * Math.Tan(Math.PI * w / (2 * fs))).ToArray();
            return wn.Select(w => 2 * fs * Math.Tan(Math.PI * w / 2)).ToArray();

        }

        /// <summary>
        /// 根据滤波器类型进行频率变换
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) TransformFilter(Complex[] z, Complex[] p, double k, string btype, double[] warped, bool analog)
        {
            return btype switch
            {
                "lowpass" => Lp2Lp(z, p, k, warped[0]),
                "highpass" => Lp2Hp(z, p, k, warped[0]),
                "bandpass" => Lp2Bp(z, p, k, warped[0], warped[1]),
                "bandstop" => Lp2Bs(z, p, k, warped[0], warped[1]),
                _ => throw new ArgumentException($"无效的滤波器类型: {btype}")
            };
        }

        /// <summary>
        /// 低通到低通变换
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) Lp2Lp(Complex[] z, Complex[] p, double k, double wo)
        {
            var zNew = z.Select(zz => zz * wo).ToArray();
            var pNew = p.Select(pp => pp * wo).ToArray();
            double kNew = k * Math.Pow(wo, p.Length - z.Length);
            //double kNew = 0.6509539939760569;// k * Math.Pow(wo, z.Length - p.Length);
            return (zNew, pNew, kNew);
        }

        /// <summary>
        /// 低通到高通变换
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) Lp2Hp(Complex[] z, Complex[] p, double k, double wo)
        {
            var zNew = z.Length > 0 ? z.Select(zz => wo / zz).ToArray() : Array.Empty<Complex>();
            var pNew = p.Select(pp => wo / pp).ToArray();

            // 补充零点使零点数等于极点数
            int zerosToAdd = p.Length - z.Length;
            zNew = zNew.Concat(Enumerable.Repeat(Complex.Zero, zerosToAdd)).ToArray();

            double kNew = k * Math.Pow(wo, z.Length - p.Length);
            return (zNew, pNew, kNew);
        }

        /// <summary>
        /// 低通到带通变换
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) Lp2Bp(Complex[] z, Complex[] p, double k, double w1, double w2)
        {
            double wo = Math.Sqrt(w1 * w2); // 中心频率
            double bw = w2 - w1; // 带宽

            var zNew = new List<Complex>();
            foreach (var zero in z)
            {
                if (zero.Magnitude < 1e-10) // 原点零点 → ±j*wo
                {
                    zNew.Add(Complex.ImaginaryOne * wo);
                    zNew.Add(-Complex.ImaginaryOne * wo);
                }
                else // 其他零点 → 共轭对
                {
                    double b = -zero.Real * bw;
                    double discriminant = b * b - 4 * wo * wo;

                    if (discriminant >= 0)
                    {
                        double sqrtD = Math.Sqrt(discriminant);
                        zNew.Add(new Complex((-b + sqrtD) / 2, 0));
                        zNew.Add(new Complex((-b - sqrtD) / 2, 0));
                    }
                    else
                    {
                        double realPart = -b / 2;
                        double imagPart = Math.Sqrt(-discriminant) / 2;
                        zNew.Add(new Complex(realPart, imagPart));
                        zNew.Add(new Complex(realPart, -imagPart));
                    }
                }
            }

            var pNew = new List<Complex>();
            foreach (var pole in p)
            {
                double b = -pole.Real * bw;
                double discriminant = b * b - 4 * wo * wo;

                if (discriminant >= 0)
                {
                    double sqrtD = Math.Sqrt(discriminant);
                    pNew.Add(new Complex((-b + sqrtD) / 2, 0));
                    pNew.Add(new Complex((-b - sqrtD) / 2, 0));
                }
                else
                {
                    double realPart = -b / 2;
                    double imagPart = Math.Sqrt(-discriminant) / 2;
                    pNew.Add(new Complex(realPart, imagPart));
                    pNew.Add(new Complex(realPart, -imagPart));
                }
            }

            double kNew = k * Math.Pow(bw, z.Length - p.Length);
            return (zNew.ToArray(), pNew.ToArray(), kNew);
        }

        /// <summary>
        /// 低通到带阻变换
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) Lp2Bs(Complex[] z, Complex[] p, double k, double w1, double w2)
        {
            double wo = Math.Sqrt(w1 * w2);
            double bw = w2 - w1;

            var zNew = new List<Complex>();
            foreach (var zero in z)
            {
                if (zero.Magnitude < 1e-10) // 原点零点 → ±j*wo
                {
                    zNew.Add(Complex.ImaginaryOne * wo);
                    zNew.Add(-Complex.ImaginaryOne * wo);
                }
                else // 其他零点 → 共轭对
                {
                    double denom = zero.Real * bw;
                    zNew.Add(new Complex(0, wo * wo / denom));
                    zNew.Add(new Complex(0, -wo * wo / denom));
                }
            }

            var pNew = new List<Complex>();
            foreach (var pole in p)
            {
                double denom = pole.Real * bw;
                pNew.Add(new Complex(0, wo * wo / denom));
                pNew.Add(new Complex(0, -wo * wo / denom));
            }

            double kNew = k * Math.Pow(wo, z.Length - p.Length);
            return (zNew.ToArray(), pNew.ToArray(), kNew);
        }

        /// <summary>
        /// 双线性变换（从s域到z域）
        /// </summary>
        private static (Complex[] z, Complex[] p, double k) BilinearZpk(Complex[] z, Complex[] p, double k, double fs)
        {
            // 修正1：T应为采样周期（1/fs），而非2/fs
            double T = 1.0 / fs;  // 正确定义：采样周期T = 1/fs
            double sScaling = T / 2;  // s的缩放系数：T/2 = 1/(2fs)，符合双线性变换公式

            // 修正2：数字域极点/零点计算（严格匹配公式z = (1 + s/(2fs)) / (1 - s/(2fs))）
            var zDigital = z.Select(zz => (1 + zz * sScaling) / (1 - zz * sScaling)).ToArray();
            var pDigital = p.Select(pp => (1 + pp * sScaling) / (1 - pp * sScaling)).ToArray();

            // 补充零点（确保零点数=极点数，Butterworth需补z=-1）
            int diff = p.Length - z.Length;
            if (diff > 0)
                zDigital = zDigital.Concat(Enumerable.Repeat(new Complex(-1, 0), diff)).ToArray();

            //// 修正3：增益调整的乘积项（匹配SciPy的prodP和prodZ计算）
            //Complex prodP = Complex.One;
            //foreach (var pp in p) prodP *= (1 - pp * sScaling);  // 对应(1 - s/(2fs))的乘积
            //Complex prodZ = Complex.One;
            ////foreach (var zz in z) prodZ *= (1 - zz * sScaling);

            //// 增益修正（使用实部，与SciPy一致）
            ////k *= (prodP / prodZ).Real;

            //// 当模拟域无零点时（如Butterworth），跳过prodZ计算
            //double gainAdjustment = z.Length == 0
            //    ? prodP.Real  // SciPy特殊处理：无零点时直接使用prodP的实部
            //    : (prodP / prodZ).Real;

            //k *= gainAdjustment;

            // 计算原始极点和零点的乘积项（用于增益调整）
            Complex prodP = Complex.One;
            foreach (var pp in p)
                prodP *= (1 - pp * sScaling);

            Complex prodZ = Complex.One;
            foreach (var zz in z)
                prodZ *= (1 - zz * sScaling);

            // 增益调整（关键修正：考虑补充零点在DC处的响应）
            double dcGain = Math.Pow(2, diff);  // 补充零点z=-1在DC处(z=1)的增益贡献
            k *= (prodP / prodZ).Real / dcGain;

            return (zDigital, pDigital, k);
        }

        /// <summary>
        /// 计算归一化增益（确保通带增益为1）
        /// </summary>        
        private static double ComputeNormalizationGain(Complex[] z, Complex[] p, double k, string btype)
        {
            Complex evalPoint = (btype == "lowpass" || btype == "bandstop")
                ? Complex.One
                : new Complex(-1, 0);

            // 精确计算增益（保留复数相位，避免仅用模长导致的误差）
            Complex numerator = Complex.One;
            foreach (var zero in z) numerator *= (evalPoint - zero);

            Complex denominator = Complex.One;
            foreach (var pole in p) denominator *= (evalPoint - pole);

            Complex gainComplex = k * (numerator / denominator);
            return gainComplex.Magnitude; // 用模长确保增益为正
        }

        /// <summary>
        /// 将ZPK表示转换为分子/分母多项式表示
        /// </summary>
        private static (double[] b, double[] a) ZpkToBa(Complex[] z, Complex[] p, double k)
        {
            // 处理零点（保持不变，因b值正确）
            double[] b = { k };
            for (int i = 0; i < z.Length; i++)
            {
                var z1 = z[i];
                if (i + 1 < z.Length && AreConjugates(z1, z[i + 1])) // 改用容差判断
                {
                    double re = z1.Real;
                    double im = z1.Imaginary;
                    b = MultiplyPolynomial(b, new[] { 1.0, -2 * re, re * re + im * im });
                    i++;
                }
                else
                {
                    b = MultiplyPolynomial(b, new[] { 1.0, -z1.Real });
                }
            }

            // 处理极点（核心修正）
            double[] a = { 1.0 };
            var processed = new bool[p.Length]; // 标记已处理的极点

            for (int i = 0; i < p.Length; i++)
            {
                if (processed[i]) continue; // 跳过已处理的极点
                var p1 = p[i];

                // 查找共轭极点（不局限于相邻项）
                int conjugateIndex = -1;
                for (int j = i + 1; j < p.Length; j++)
                {
                    if (!processed[j] && AreConjugates(p1, p[j]))
                    {
                        conjugateIndex = j;
                        break;
                    }
                }

                if (conjugateIndex != -1)
                {
                    // 合并共轭极点对
                    double re = p1.Real;
                    double im = p1.Imaginary;
                    // 计算二次项系数（关键：保留更多小数位精度）
                    double coeff2 = 1.0;
                    double coeff1 = -2 * re;
                    double coeff0 = re * re + im * im; // 模长平方
                    a = MultiplyPolynomial(a, new[] { coeff2, coeff1, coeff0 });
                    processed[i] = true;
                    processed[conjugateIndex] = true;
                }
                else
                {
                    // 处理实极点（虚部接近0）
                    a = MultiplyPolynomial(a, new[] { 1.0, -p1.Real });
                    processed[i] = true;
                }
            }

            // 归一化分母首项为1（保持不变）
            double a0 = a[0];
            b = b.Select(x => x / a0).ToArray();
            a = a.Select(x => x / a0).ToArray();

            return (b, a);
        }

        // 辅助：用容差判断两个复数是否为共轭对（核心修正）
        private static bool AreConjugates(Complex c1, Complex c2)
        {
            const double tolerance = 1e-10; // 允许的浮点误差
                                            // 实部应相等，虚部应互为相反数
            return Math.Abs(c1.Real - c2.Real) < tolerance
                && Math.Abs(c1.Imaginary + c2.Imaginary) < tolerance;
        }

        // 优化多项式乘法（减少累积误差）
        private static double[] MultiplyPolynomial(double[] a, double[] b)
        {
            int len = a.Length + b.Length - 1;
            double[] result = new double[len];
            // 交换乘法顺序：用短多项式乘以长多项式，减少循环次数
            if (a.Length > b.Length)
            {
                var temp = a;
                a = b;
                b = temp;
            }
            // 逐元素相乘，用临时变量存储中间结果，减少访问冲突
            for (int i = 0; i < a.Length; i++)
            {
                double ai = a[i];
                for (int j = 0; j < b.Length; j++)
                {
                    result[i + j] += ai * b[j];
                }
            }
            return result;
        }

        public static double[] FiltFilt(double[] b, double[] a, double[] x)
        {
            if (b == null || b.Length == 0)
                throw new ArgumentException("Numerator coefficients (b) cannot be empty.", nameof(b));

            if (a == null || a.Length == 0)
                throw new ArgumentException("Denominator coefficients (a) cannot be empty.", nameof(a));

            if (x == null || x.Length == 0)
                return new double[0];

            // Normalize coefficients if a[0] != 1
            if (a[0] != 1.0)
            {
                double a0 = a[0];
                b = b.Select(bi => bi / a0).ToArray();
                a = a.Select(ai => ai / a0).ToArray();
            }

            // Determine padding length (default: 3*max(len(a),len(b)))
            int padlen = 3 * Math.Max(a.Length, b.Length);
            padlen = Math.Min(padlen, x.Length - 1);

            // Use "odd" padding (reflect)
            double[] padded = PadSignal(x, padlen, "odd");

            // Forward filter
            double[] yForward = Filter(b, a, padded);

            // Reverse the filtered signal
            double[] yReversed = yForward.Reverse().ToArray();

            // Backward filter
            double[] yBackward = Filter(b, a, yReversed);

            // Reverse back and extract the central part
            double[] result = yBackward.Reverse().ToArray();
            double[] output = new double[x.Length];
            Array.Copy(result, padlen, output, 0, x.Length);

            return output;
        }

        private static double[] PadSignal(double[] x, int padlen, string padtype)
        {
            if (padlen <= 0)
                return (double[])x.Clone();

            double[] padded = new double[x.Length + 2 * padlen];

            // Central part is the original signal
            Array.Copy(x, 0, padded, padlen, x.Length);

            // Handle padding based on padtype
            switch (padtype.ToLower())
            {
                case "odd":
                    // Odd symmetry: x[-n] = 2x[0] - x[n]
                    for (int i = 0; i < padlen; i++)
                    {
                        padded[padlen - 1 - i] = 2 * x[0] - x[Math.Min(i + 1, x.Length - 1)];
                        padded[padded.Length - padlen + i] = 2 * x[x.Length - 1] - x[Math.Max(x.Length - 2 - i, 0)];
                    }
                    break;

                case "even":
                    // Even symmetry: x[-n] = x[n]
                    for (int i = 0; i < padlen; i++)
                    {
                        padded[padlen - 1 - i] = x[Math.Min(i + 1, x.Length - 1)];
                        padded[padded.Length - padlen + i] = x[Math.Max(x.Length - 2 - i, 0)];
                    }
                    break;

                case "constant":
                    // Constant padding: x[-n] = x[0]
                    for (int i = 0; i < padlen; i++)
                    {
                        padded[padlen - 1 - i] = x[0];
                        padded[padded.Length - padlen + i] = x[x.Length - 1];
                    }
                    break;

                default:
                    throw new ArgumentException("Unsupported padtype. Use 'odd', 'even', or 'constant'.", nameof(padtype));
            }

            return padded;
        }

        private static double[] Filter(double[] b, double[] a, double[] x)
        {
            // Direct Form II Transposed implementation of IIR filter
            double[] y = new double[x.Length];
            double[] z = new double[a.Length - 1]; // State vector

            for (int n = 0; n < x.Length; n++)
            {
                // Compute output
                y[n] = b[0] * x[n] + z[0];

                // Update state
                for (int i = 0; i < z.Length - 1; i++)
                {
                    z[i] = b[i + 1] * x[n] + z[i + 1] - a[i + 1] * y[n];
                }

                if (z.Length > 0)
                {
                    z[z.Length - 1] = b[b.Length - 1] * x[n] - a[a.Length - 1] * y[n];
                }
            }

            return y;
        }
        ///////////////////////////////////////

        /// <summary>
        /// 计算信号的短时傅里叶变换（STFT），类似于scipy.signal.stft的功能
        /// </summary>
        /// <param name="signal">输入的时域信号，一维数组形式</param>
        /// <param name="fs">采样频率，单位Hz</param>
        /// <param name="windowLength">窗函数长度，默认2048，建议是2的幂次方以提高FFT效率</param>
        /// <param name="hopLength">帧移长度，默认512</param>
        /// <param name="windowFunction">窗函数类型，目前支持"Hann"，可扩展支持更多类型，默认"Hann"</param>
        /// <returns>返回包含频率轴、时间轴和复数频谱矩阵的元组，对应Python中stft返回的f, t, Zxx</returns>
        public static (double[] frequencies, double[] times, Matrix<Complex> spectrogram) Stft(
        double[] x,
        double fs = 1.0,
        string window = "hann",
        int nperseg = 256,
        int? noverlap = null,
        int? nfft = null,
        bool detrend = false,
        bool return_onesided = true,
        string boundary = "zeros",
        bool padded = true,
        string scaling = "spectrum")
        {
            // 参数验证
            if (x == null || x.Length == 0)
                throw new ArgumentException("Input signal cannot be empty.");

            if (fs <= 0)
                throw new ArgumentException("Sampling frequency must be positive.");

            if (nperseg <= 0)
                throw new ArgumentException("Segment length must be positive.");

            noverlap = noverlap ?? nperseg / 2;
            if (noverlap < 0 || noverlap >= nperseg)
                throw new ArgumentException("noverlap must be between 0 and nperseg-1.");

            nfft = nfft ?? nperseg;
            if (nfft < nperseg)
                throw new ArgumentException("nfft must be greater than or equal to nperseg.");

            // 边界扩展
            double[] xExtended = ApplyBoundaryExtension(x, nperseg, boundary);
            int originalLength = x.Length;
            int extendedLength = xExtended.Length;

            // 计算实际使用的信号长度
            int step = nperseg - noverlap.Value;
            int numSegments = (int)Math.Ceiling((extendedLength - nperseg) / (double)step) + 1;

            // 如果需要填充
            if (padded)
            {
                int requiredLength = (numSegments - 1) * step + nperseg;
                if (extendedLength < requiredLength)
                {
                    Array.Resize(ref xExtended, requiredLength);
                    extendedLength = requiredLength;
                }
            }

            // 窗函数
            double[] win = GenerateWindow(window, nperseg);
            double winSum = win.Sum();
            double winSumSquared = win.Select(w => w * w).Sum();

            // 计算缩放因子
            double scale = scaling == "psd"
                ? 1.0 / (fs * winSumSquared)
                : 1.0 / winSum;

            // 计算频率轴 (修正部分)
            double[] frequencies = GenerateFrequencies(nfft.Value, fs, return_onesided);

            //// 计算时间轴 (修正部分)
            //double[] times = new double[numSegments];
            //for (int i = 0; i < numSegments; i++)
            //{
            //    // 时间点对应原始信号中的中心位置
            //    int centerPos = i * step + nperseg / 2;
            //    // 减去边界扩展的长度，映射回原始信号时间
            //    double adjustedPos = centerPos - (extendedLength - originalLength) / 2;
            //    times[i] = adjustedPos / fs;
            //}

            // 初始化频谱图矩阵
            int numFreqBins = return_onesided ? (nfft.Value / 2 + 1) : nfft.Value;
            var spectrogram = Matrix<Complex>.Build.Dense(numFreqBins, numSegments);

            // 处理每个段
            for (int seg = 0; seg < numSegments; seg++)
            {
                int start = seg * step;

                // 提取段数据
                double[] segment = new double[nperseg];
                Array.Copy(xExtended, start, segment, 0, nperseg);

                // 去趋势
                if (detrend)
                {
                    segment = Detrend(segment);
                }

                // 加窗
                for (int i = 0; i < nperseg; i++)
                {
                    segment[i] *= win[i];
                }

                // 零填充和FFT
                Complex[] fftInput = new Complex[nfft.Value];
                for (int i = 0; i < nperseg; i++)
                {
                    fftInput[i] = new Complex(segment[i], 0);
                }

                Fourier.Forward(fftInput, FourierOptions.AsymmetricScaling);

                // 存储结果并应用缩放
                if (return_onesided)
                {
                    for (int i = 0; i < numFreqBins; i++)
                    {
                        spectrogram[i, seg] = fftInput[i] * scale;
                    }
                }
                else
                {
                    for (int i = 0; i < nfft.Value; i++)
                    {
                        int idx = i < numFreqBins ? i : i - nfft.Value;
                        spectrogram[idx, seg] = fftInput[i] * scale;
                    }
                }
            }

            double[] times = GenerateTimes(numSegments, step, fs);

            return (frequencies, times, spectrogram);
        }

        private static double[] GenerateFrequencies(int nfft, double fs, bool return_onesided)
        {
            int numBins = return_onesided ? (nfft / 2 + 1) : nfft;
            double[] freqs = new double[numBins];
            double deltaF = fs / nfft;

            for (int i = 0; i < numBins; i++)
            {
                freqs[i] = i * deltaF;
            }

            if (!return_onesided)
            {
                for (int i = numBins / 2 + 1; i < numBins; i++)
                {
                    freqs[i] -= fs;
                }
            }

            return freqs;
        }

        private static double[] ApplyBoundaryExtension(double[] x, int nperseg, string boundary)
        {
            if (boundary == null || boundary.ToLower() == "none")
                return (double[])x.Clone();

            int extLen = nperseg / 2;
            if (extLen <= 0)
                return (double[])x.Clone();

            double[] extended = new double[x.Length + 2 * extLen];

            // Copy original data to the center
            Array.Copy(x, 0, extended, extLen, x.Length);

            switch (boundary.ToLower())
            {
                case "zeros":
                    // Already zero-padded by array initialization
                    break;

                case "even":
                    // Even symmetry (mirror reflection)
                    for (int i = 0; i < extLen; i++)
                    {
                        extended[i] = x[1 + i];  // Start from x[1] to avoid duplicating x[0]
                        extended[extLen + x.Length + i] = x[x.Length - 2 - i];
                    }
                    break;

                case "odd":
                    // Odd symmetry (mirror reflection with sign flip)
                    for (int i = 0; i < extLen; i++)
                    {
                        extended[i] = 2 * x[0] - x[1 + i];
                        extended[extLen + x.Length + i] = 2 * x[x.Length - 1] - x[x.Length - 2 - i];
                    }
                    break;

                case "constant":
                    // Constant extension
                    double first = x[0];
                    double last = x[x.Length - 1];
                    for (int i = 0; i < extLen; i++)
                    {
                        extended[i] = first;
                        extended[extLen + x.Length + i] = last;
                    }
                    break;

                default:
                    throw new ArgumentException($"Invalid boundary option: {boundary}");
            }

            return extended;
        }

        private static double[] GenerateWindow(string windowType, int nperseg, bool periodic = true)
        {
            switch (windowType.ToLower())
            {
                case "hann":
                    //double[] hann = new double[nperseg];
                    //for (int i = 0; i < nperseg; i++)
                    //{
                    //    hann[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (nperseg - 1)));
                    //}
                    //return hann;
                    int windowLength = periodic ? nperseg + 1 : nperseg;
                    double[] window = new double[windowLength];

                    for (int i = 0; i < windowLength; i++)
                    {
                        double angle = 2.0 * Math.PI * i / (windowLength - 1);
                        window[i] = 0.5f * (1.0f - (float)Math.Cos(angle));
                    }

                    return periodic ? window.Take(nperseg).ToArray() : window;

                case "hamming":
                    double[] hamming = new double[nperseg];
                    for (int i = 0; i < nperseg; i++)
                    {
                        hamming[i] = 0.54 - 0.46 * Math.Cos(2 * Math.PI * i / (nperseg - 1));
                    }
                    return hamming;

                case "boxcar":
                    return new double[nperseg].Select(_ => 1.0).ToArray();

                // Add more window types as needed
                default:
                    throw new ArgumentException($"Window type '{windowType}' not supported.");
            }
        }

        private static double[] Detrend(double[] segment)
        {
            // Simple linear detrending
            double[] x = Generate.LinearRange(0, segment.Length - 1);
            (double, double) coefficients = Fit.Line(x, segment);
            double intercept = coefficients.Item1;
            double slope = coefficients.Item2;

            double[] detrended = new double[segment.Length];
            for (int i = 0; i < segment.Length; i++)
            {
                detrended[i] = segment[i] - (intercept + slope * i);
            }

            return detrended;
        }

        /// <summary>
        /// 时间轴数组
        /// </summary>
        /// <param name="numFrames">帧数</param>
        /// <param name="hopLength">帧移长度</param>
        /// <param name="fs">采样频率</param>
        /// <returns>时间轴数组，单位秒</returns>
        private static double[] GenerateTimes(int numFrames, int hopLength, double fs)
        {
            double[] times = new double[numFrames];
            for (int i = 0; i < numFrames; i++)
            {
                times[i] = (i * hopLength) / fs;
            }
            return times;
        }

        /// <summary>
        /// Compute the Inverse Short-Time Fourier Transform (ISTFT).
        /// </summary>
        /// <param name="stft">STFT matrix</param>
        /// <param name="nperseg">Length of each segment</param>
        /// <param name="noverlap">Number of points to overlap between segments</param>
        /// <param name="nfft">Length of the FFT used</param>
        /// <param name="window">Window function used in STFT</param>
        /// <returns>Reconstructed time-domain signal</returns>
        public static double[] Istft(Complex[,] stft, int nperseg = 256, int? noverlap = null, int? nfft = null, double[] window = null)
        {
            noverlap = noverlap ?? nperseg / 2;
            nfft = nfft ?? nperseg;

            if (noverlap >= nperseg)
                throw new ArgumentException("noverlap must be less than nperseg.");

            if (window == null)
            {
                window = new double[nperseg];
                for (int i = 0; i < nperseg; i++)
                    window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (nperseg - 1))); // Hann window
            }
            else if (window.Length != nperseg)
            {
                throw new ArgumentException("Window must have length nperseg.");
            }

            int step = nperseg - noverlap.Value;
            int nseg = stft.GetLength(1);
            int len = (nseg - 1) * step + nperseg;

            double[] x = new double[len];
            double[] winSum = new double[len];

            for (int seg = 0; seg < nseg; seg++)
            {
                int start = seg * step;

                // Get segment from STFT
                Complex[] segment = new Complex[nfft.Value];
                for (int i = 0; i < nfft; i++)
                    segment[i] = stft[i, seg];

                // Compute inverse FFT
                Fourier.Inverse(segment, FourierOptions.Default);

                // Apply window and overlap-add
                for (int i = 0; i < nperseg; i++)
                {
                    if (start + i < len)
                    {
                        x[start + i] += segment[i].Real * window[i];
                        winSum[start + i] += window[i] * window[i];
                    }
                }
            }

            // Normalize by window sum
            for (int i = 0; i < len; i++)
            {
                if (winSum[i] != 0)
                    x[i] /= winSum[i];
            }

            return x;
        }
    }
}