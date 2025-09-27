using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfZipEnhancerSe : ISepProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        //private int _required_cache_size = 0;
        public SepProjOfZipEnhancerSe(SepModel sepModel)
        {
            _modelSession = sepModel.ModelSession;
            _customMetadata = sepModel.CustomMetadata;
            _featureDim = sepModel.FeatureDim;
            _sampleRate = sepModel.SampleRate;
            _channels = sepModel.Channels;
            _chunkLength = sepModel.ChunkLength;
            _shiftLength = sepModel.ShiftLength;
            //_required_cache_size = sepModel.Required_cache_size;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Channels { get => _channels; set => _channels = value; }

        //public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }

        //public List<float[]> GetEncoderInitStates(int batchSize = 1)
        //{
        //    List<float[]> statesList = new List<float[]>();
        //    //计算尺寸
        //    int required_cache_size = _required_cache_size < 0 ? 0 : _required_cache_size;
        //    float[] att_cache = new float[_customMetadata.Num_blocks * _customMetadata.Head * required_cache_size * (_customMetadata.Output_size / _customMetadata.Head * 2)];
        //    float[] cnn_cache = new float[_customMetadata.Num_blocks * 1 * _customMetadata.Output_size * (_customMetadata.Cnn_module_kernel - 1)];
        //    statesList.Add(att_cache);
        //    statesList.Add(cnn_cache);
        //    return statesList;
        //}
        //private float[] InitCacheFeats(int batchSize = 1)
        //{

        //    int cached_feature_size = 0;//1 + _right_context - _subsampling_rate;//TODO temp test
        //    float[] cacheFeats = new float[batchSize * cached_feature_size * 80];
        //    return cacheFeats;
        //}

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            states = statesList[0];
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 2 == 0, "when stack_states, state_list[0] is 2x");
            statesList.Add(states);
            return statesList;
        }

        /// <summary>
        /// Converts a Complex[961, 1, 1808] array to float[961, 1808, 2] STFT format
        /// </summary>
        /// <param name="complexSpectrogram">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [freq_bins, time_frames, 2] (实部和虚部分量)</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexSpectrogram)
        {
            int freqBins = complexSpectrogram.GetLength(0);   // 961 (频率bin)
            int channels = complexSpectrogram.GetLength(1);   // 1 (单通道)
            int timeFrames = complexSpectrogram.GetLength(2); // 1808 (时间帧)

            // 验证输入维度是否符合预期
            if (channels != 1)
            {
                throw new ArgumentException($"输入的通道数必须为1，实际为{channels}", nameof(complexSpectrogram));
            }

            // 目标形状: [freq_bins, time_frames, 2] (第三维存储实部和虚部)
            float[,,] stftFormat = new float[freqBins, timeFrames, 2];

            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 提取复数的实部到第三维索引0
                    stftFormat[f, t, 0] = (float)complexSpectrogram[f, 0, t].Real;
                    // 提取复数的虚部到第三维索引1
                    stftFormat[f, t, 1] = (float)complexSpectrogram[f, 0, t].Imaginary;
                }
            }

            return stftFormat;
        }



        /// <summary>
        /// 将 float[201, 时间帧数, 1] 格式的 STFT 转换为 Complex[201, 1, 时间帧数] 格式的复数频谱
        /// </summary>
        /// <param name="stftFormat">输入的 STFT 数据，格式为 [频率点(201), 时间帧数, 实虚部(0:实部,1:虚部)]</param>
        /// <returns>复数频谱，格式为 [频率点(201), 1, 时间帧数]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            // 获取输入维度
            int freqBins = stftFormat.GetLength(0);     // 频率点数量（应等于201）
            int timeFrames = stftFormat.GetLength(1);   // 时间帧数
            int complexComponents = stftFormat.GetLength(2);  // 实虚部分量（应等于2）

            // 验证输入维度有效性
            if (freqBins != 201)
            {
                throw new ArgumentException($"输入频率点数量必须为201，实际为{freqBins}");
            }
            //if (complexComponents != 2)
            //{
            //    throw new ArgumentException($"输入实虚部分量必须为2，实际为{complexComponents}");
            //}

            // 初始化目标复数数组 [201, 1, 时间帧数]
            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            // 遍历所有频率点和时间帧，转换为复数
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 提取实部和虚部
                    float realPart = stftFormat[f, t, 0];
                    float imagPart = 0f;
                    if (complexComponents == 2)
                    {
                        imagPart = stftFormat[f, t, 1];
                    }

                    // 特殊处理Nyquist频率点（索引100）：实数信号的Nyquist频率虚部应为0
                    if (f == 100)
                    {
                        complexSpectrogram[f, 0, t] = new Complex(realPart, 0f);
                    }
                    else
                    {
                        complexSpectrogram[f, 0, t] = new Complex(realPart, imagPart);
                    }
                }
            }

            return complexSpectrogram;
        }


        /// <summary>
        /// 将 float[201, 时间帧数, 2] 格式的 STFT 转换为 Complex[201, 2, 时间帧数] 格式的复数频谱
        /// </summary>
        /// <param name="stftFormat">输入的 STFT 数据，格式为 [频率(201), 时间帧数, 实虚部(0:实部,1:虚部)]</param>
        /// <returns>复数频谱，格式为 [频率(201), 2, 时间帧数]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex2(float[,,] stftFormat)
        {
            // 获取输入维度
            int freqBins = stftFormat.GetLength(0);     // 频率点数量（应等于201）
            int timeFrames = stftFormat.GetLength(1);   // 时间帧数（如427）
            int complexComponents = stftFormat.GetLength(2);  // 实虚部分量（应等于2）

            // 验证输入维度有效性
            if (freqBins != 201)
            {
                throw new ArgumentException($"输入频率点数量必须为201，实际为{freqBins}");
            }
            if (complexComponents != 2)
            {
                throw new ArgumentException($"输入实虚部分量必须为2，实际为{complexComponents}");
            }

            // 创建目标复数数组 [201, 2, 时间帧数]（匹配注释要求的输出格式）
            Complex[,,] complexSpectrogram = new Complex[freqBins, 2, timeFrames];

            // 遍历所有频率点和时间帧，转换为复数并填充到二维通道
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 提取实部和虚部
                    float realPart = stftFormat[f, t, 0];
                    float imagPart = stftFormat[f, t, 1];

                    // 特殊处理Nyquist频率点（索引100）：实数信号的Nyquist频率虚部应为0
                    if (f == 100)
                    {
                        imagPart = 0f;
                    }

                    // 构建复数（实部+虚部）
                    Complex complexValue = new Complex(realPart, imagPart);

                    // 填充输出数组的两个通道（根据业务需求，此处均使用相同复数，可按需修改）
                    complexSpectrogram[f, 0, t] = complexValue;
                    complexSpectrogram[f, 1, t] = complexValue;
                }
            }

            return complexSpectrogram;
        }


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

            //Array.Copy(sample, 0, leftChannel, 0, channelLength);
            //Array.Copy(sample, channelLength, rightChannel, 0, channelLength);

            for (int n = 0; n < channelLength; n++)
            {
                leftChannel[n] = sample[n * 2];
                rightChannel[n] = sample[n * 2 + 1];
            }

            return (leftChannel, rightChannel);
        }

        public static float[]? MergeMonoToStereo(float[] leftChannel, float[] rightChannel)
        {
            // 验证输入有效性
            if (leftChannel == null || rightChannel == null)
            {
                Console.WriteLine("错误: 左右声道数据不能为空");
                return null;
            }

            if (leftChannel.Length != rightChannel.Length)
            {
                Console.WriteLine($"错误: 左右声道样本数不一致（左: {leftChannel.Length}, 右: {rightChannel.Length}）");
                return null;
            }

            int stereoLength = leftChannel.Length * 2;
            float[] stereoSamples = new float[stereoLength];

            // 合并左右声道数据
            for (int i = 0; i < leftChannel.Length; i++)
            {
                stereoSamples[i * 2] = leftChannel[i];         // 左声道数据放在偶数索引位置
                stereoSamples[i * 2 + 1] = rightChannel[i];    // 右声道数据放在奇数索引位置
            }

            return stereoSamples;
        }

        public static float GetNormFactor(float[] noisyWav)
        {
            if (noisyWav == null || noisyWav.Length == 0)
                throw new ArgumentException("输入音频数据不能为空或空数组");

            // 计算音频长度（对应Python中的noisy_wav.shape[1]）
            int length = noisyWav.Length;

            // 计算音频平方和（对应Python中的torch.sum(noisy_wav ** 2.0)）
            float sumOfSquares = 0f;
            foreach (float sample in noisyWav)
            {
                sumOfSquares += sample * sample;
            }

            // 处理平方和为0的特殊情况，避免除零错误
            if (sumOfSquares < 1e-10f)
            {
                return 0f; // 返回全零数组
            }

            // 计算归一化因子（对应Python中的norm_factor）
            float normFactor = (float)Math.Sqrt(length / sumOfSquares);
            return normFactor;
        }

        /// <summary>
        /// 对音频采样数据进行归一化处理
        /// 等效于Python中的归一化逻辑
        /// </summary>
        /// <param name="noisyWav">音频采样数据，float数组</param>
        /// <returns>归一化后的音频数据</returns>
        public static (float[] normalizedAudio, float normFactor) NormalizeAudio(float[] noisyWav)
        {
            if (noisyWav == null || noisyWav.Length == 0)
                throw new ArgumentException("输入音频数据不能为空或空数组");

            // 计算音频长度（对应Python中的noisy_wav.shape[1]）
            int length = noisyWav.Length;

            // 计算音频平方和（对应Python中的torch.sum(noisy_wav ** 2.0)）
            float sumOfSquares = 0f;
            foreach (float sample in noisyWav)
            {
                sumOfSquares += sample * sample;
            }

            // 处理平方和为0的特殊情况，避免除零错误
            if (sumOfSquares < 1e-10f)
            {
                return (new float[length], 0f); // 返回全零数组
            }

            // 计算归一化因子（对应Python中的norm_factor）
            float normFactor = (float)Math.Sqrt(length / sumOfSquares);

            // 应用归一化因子（对应Python中的noisy_audio = (noisy_wav * norm_factor)）
            float[] normalizedAudio = new float[length];
            for (int i = 0; i < length; i++)
            {
                normalizedAudio[i] = noisyWav[i] * normFactor;
            }

            return (normalizedAudio, normFactor);
        }

        /// <summary>
        /// 处理STFT频谱，计算幅度谱、相位谱并进行幅度压缩
        /// 等效于Python中的get_mag_pha函数，返回值为元组(mag, pha, com)
        /// </summary>
        /// <param name="stftSpec">输入的STFT频谱，三维数组[freq_bins, time_frames, 2]，最后一维为实部(0)和虚部(1)</param>
        /// <param name="compressFactor">幅度压缩因子</param>
        /// <returns>元组包含三个元素：
        /// 1. 幅度谱 [freq_bins, time_frames]
        /// 2. 相位谱 [freq_bins, time_frames]
        /// 3. 组合结果 [freq_bins, time_frames, 2]（实部和虚部）
        /// </returns>
        public static (float[,] mag, float[,] pha, float[,,] com) GetMagPha(float[,,] stftSpec, float compressFactor)
        {
            // 获取输入维度
            int freqBins = stftSpec.GetLength(0);
            int timeFrames = stftSpec.GetLength(1);
            int components = stftSpec.GetLength(2);

            // 验证输入维度（第三维必须为2，包含实部和虚部）
            if (components != 2)
            {
                throw new ArgumentException($"输入数组第三维必须为2（实部和虚部），实际为{components}", nameof(stftSpec));
            }

            // 初始化结果数组
            float[,] mag = new float[freqBins, timeFrames];
            float[,] pha = new float[freqBins, timeFrames];
            float[,,] com = new float[freqBins, timeFrames, 2];

            // 处理每个频率点和时间帧
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 提取实部和虚部
                    float real = stftSpec[f, t, 0];
                    float imag = stftSpec[f, t, 1];

                    // 计算幅度谱：sqrt(real² + imag² + 1e-9)
                    mag[f, t] = (float)Math.Sqrt(real * real + imag * imag + 1e-9f);

                    // 计算相位谱：atan2(imag, real + 1e-5)
                    pha[f, t] = (float)Math.Atan2(imag, real + 1e-5f);

                    // 幅度压缩：mag^compressFactor
                    float compressedMag = (float)Math.Pow(mag[f, t], compressFactor);

                    mag[f, t] = compressedMag;

                    // 计算组合结果的实部和虚部
                    com[f, t, 0] = compressedMag * (float)Math.Cos(pha[f, t]);
                    com[f, t, 1] = compressedMag * (float)Math.Sin(pha[f, t]);
                }
            }

            return (mag, pha, com);
        }

        /// <summary>
        /// 将幅度谱和相位谱组合为复数形式，应用幅度压缩的逆操作
        /// 等效于Python中的getCom函数
        /// </summary>
        /// <param name="mag">幅度谱，三维数组[1, 201, 412]</param>
        /// <param name="pha">相位谱，三维数组[1, 201, 412]</param>
        /// <param name="compressFactor">压缩因子，默认值为0.3</param>
        /// <returns>复数形式的组合结果，四维数组[1, 201, 412, 1]，最后一维为实部</returns>
        public static float[,,,] GetCom(float[,,] mag, float[,,] pha, float compressFactor = 0.3f)
        {
            // 验证输入维度是否完全匹配
            if (mag.GetLength(0) != pha.GetLength(0) ||
                mag.GetLength(1) != pha.GetLength(1) ||
                mag.GetLength(2) != pha.GetLength(2))
            {
                throw new ArgumentException("幅度谱(mag)和相位谱(pha)的维度必须完全一致");
            }

            // 获取输入维度信息：[batch, 频率点, 时间帧]
            int batchSize = mag.GetLength(0);    // 对应注释中的1
            int freqBins = mag.GetLength(1);     // 对应注释中的201
            int timeFrames = mag.GetLength(2);   // 对应注释中的412

            // 初始化输出数组：[batch, 频率点, 时间帧, 1]，最后一维存储实部
            float[,,,] com = new float[batchSize, freqBins, timeFrames, 1];

            // 计算幅度压缩的逆因子（1 / 压缩因子）
            float inverseFactor = 1.0f / compressFactor;

            // 遍历所有元素执行计算
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // 对幅度谱应用逆压缩：mag^(1/compressFactor)
                        float scaledMag = (float)Math.Pow(mag[b, f, t], inverseFactor);

                        // 计算实部：scaledMag * cos(相位)，存入输出数组的实部位置
                        com[b, f, t, 0] = scaledMag * (float)Math.Cos(pha[b, f, t]);
                    }
                }
            }

            return com;
        }

        /// <summary>
        /// 将幅度谱和相位谱组合为复数形式，应用幅度压缩的逆操作
        /// 等效于Python中的getCom函数
        /// </summary>
        /// <param name="mag">幅度谱，三维数组[1, 201, 412]</param>
        /// <param name="pha">相位谱，三维数组[1, 201, 412]</param>
        /// <param name="compressFactor">压缩因子，默认值为0.3</param>
        /// <returns>复数形式的组合结果，四维数组[1, 201, 412, 2]，最后一维为实部和虚部</returns>
        public static float[,,,] GetCom2(float[,,] mag, float[,,] pha, float compressFactor = 0.3f)
        {
            // 验证输入维度是否匹配
            if (mag.GetLength(0) != pha.GetLength(0) ||
                mag.GetLength(1) != pha.GetLength(1) ||
                mag.GetLength(2) != pha.GetLength(2))
            {
                throw new ArgumentException("幅度谱和相位谱的维度不匹配");
            }

            // 获取输入维度 [batch, freq_bins, time_frames]
            int batchSize = mag.GetLength(0);
            int freqBins = mag.GetLength(1);
            int timeFrames = mag.GetLength(2);

            // 创建输出数组 [batch, freq_bins, time_frames, 2]，最后一维存储实部(0)和虚部(1)
            float[,,,] com = new float[batchSize, freqBins, timeFrames, 2];

            // 计算逆压缩因子
            float inverseFactor = 1.0f / compressFactor;

            // 遍历所有元素计算结果
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // 对幅度谱应用逆压缩: mag^(1/compress_factor)
                        float scaledMag = (float)Math.Pow(mag[b, f, t], inverseFactor);

                        // 计算实部: scaledMag * cos(pha)
                        com[b, f, t, 0] = scaledMag * (float)Math.Cos(pha[b, f, t]);

                        // 计算虚部: scaledMag * sin(pha)
                        com[b, f, t, 1] = scaledMag * (float)Math.Sin(pha[b, f, t]);
                    }
                }
            }

            return com;
        }

        /// <summary>
        /// 从四维数组中提取指定批次的数据，并转换为[freq_bins, time_frames, 1]格式
        /// </summary>
        /// <param name="com">输入的四维数组，结构为[batch, freq_bins, time_frames, components]</param>
        /// <param name="b">要提取的批次索引</param>
        /// <returns>三维数组，结构为[freq_bins, time_frames, 1]</returns>
        public static float[,,] GetBatch(float[,,,] com, int b)
        {
            // 获取输入数组的维度信息
            int batchSize = com.GetLength(0);
            int freqBins = com.GetLength(1);
            int timeFrames = com.GetLength(2);
            int components = com.GetLength(3);

            // 验证批次索引是否有效
            if (b < 0 || b >= batchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(b),
                    $"批次索引{b}超出有效范围[0, {batchSize - 1}]");
            }

            // 创建输出数组，维度为[freq_bins, time_frames, 1]
            float[,,] spectrum = new float[freqBins, timeFrames, 1];

            // 提取第b批次的数据，默认取components中的第一个分量(索引0)
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 这里取components的第0个元素，如需其他分量可修改索引
                    spectrum[f, t, 0] = com[b, f, t, 0];
                }
            }

            return spectrum;
        }

        /// <summary>
        /// 从四维数组中提取指定批次的数据，并转换为[freq_bins, time_frames, 2]格式
        /// </summary>
        /// <param name="com">输入的四维数组，结构为[batch, freq_bins, time_frames, components]</param>
        /// <param name="b">要提取的批次索引</param>
        /// <returns>三维数组，结构为[freq_bins, time_frames, 2]</returns>
        public static float[,,] GetBatch2(float[,,,] com, int b)
        {
            // 获取输入数组的维度信息
            int batchSize = com.GetLength(0);
            int freqBins = com.GetLength(1);
            int timeFrames = com.GetLength(2);
            int components = com.GetLength(3);

            // 验证批次索引和分量数量是否有效
            if (b < 0 || b >= batchSize)
            {
                throw new ArgumentOutOfRangeException(nameof(b),
                    $"批次索引{b}超出有效范围[0, {batchSize - 1}]");
            }
            if (components < 2)
            {
                throw new ArgumentException($"输入数组的分量维度必须至少为2，实际为{components}", nameof(com));
            }

            // 创建输出数组，维度为[freq_bins, time_frames, 2]
            float[,,] spectrum = new float[freqBins, timeFrames, 2];

            // 提取第b批次的数据，包含components中的前2个分量(索引0和1)
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 提取第0个分量（通常对应实部）
                    spectrum[f, t, 0] = com[b, f, t, 0];
                    // 提取第1个分量（通常对应虚部）
                    spectrum[f, t, 1] = com[b, f, t, 1];
                }
            }

            return spectrum;
        }


        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            /////////////////////////////
            //var normResult = NormalizeAudio(leftChannel);
            //leftChannel = normResult.normalizedAudio;
            //float normFactor = normResult.normFactor;
            ///////////////////////////
            float normFactor = GetNormFactor(samples);
            float[] features = samples.Select(x => x * normFactor).ToArray();
            Utils.STFTArgs args = new Utils.STFTArgs();
            args.win_len = 400;
            args.fft_len = 400;
            args.win_type = "hanning";
            args.win_inc = 100;
            // 对音频进行 STFT 变换
            Complex[,,] stftComplex = AudioProcessing.Stft(features, args, normalized: false, pad_mode: "reflect", center: true);
            float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
            var magPha = GetMagPha(stftSpec: spectrum, compressFactor: 0.3f);
            float[] noisy_pha = magPha.pha.Cast<float>().ToArray();
            float[] noisy_mag = magPha.mag.Cast<float>().ToArray();

            var inputMeta = _modelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "noisy_pha")
                {
                    int[] dim = new int[] { batchSize, 201, noisy_pha.Length / batchSize / 201 };
                    var tensor = new DenseTensor<float>(noisy_pha, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "noisy_mag")
                {
                    int[] dim = new int[] { batchSize, 201, noisy_mag.Length / batchSize / 201 };
                    var tensor = new DenseTensor<float>(noisy_mag, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _modelSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    Tensor<float> mag = encoderResults[0].AsTensor<float>();//amp_g 
                    Tensor<float> pha = encoderResults[1].AsTensor<float>();//pha_g 
                    var mag_arr = To3DArray(mag);
                    var pha_arr = To3DArray(pha);
                    var com = GetCom2(mag_arr, pha_arr, compressFactor: 0.3f);
                    var spec = GetBatch2(com: com, b: 0);//2
                    Complex[,,] spectrumX = ConvertSTFTFormatToComplex(spec);//2
                    float[] output1 = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false, center: true);
                    output1 = output1.Select(x => x / normFactor).ToArray();
                    int sampleRate = modelInputs[0].SampleRate;
                    int channels = modelInputs[0].Channels;
                    float[] output = new float[(int)(samples.Length - sampleRate * channels * 0.1f) / channels];
                    Array.Copy(output1, 0, output, 0, output.Length);
                    ModelOutputEntity modelOutput = new ModelOutputEntity();
                    modelOutput.StemName = "vocals";
                    modelOutput.StemContents = output;
                    modelOutputEntities.Add(modelOutput);

                }
            }
            catch (Exception ex)
            {
                //
            }

            return modelOutputEntities;
        }

        public List<ModelOutputEntity> ModelProj_stereo(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            /////////////////////////////
            var splitResult = SplitStereoToMono(samples);
            if (splitResult.HasValue)
            {
                float[] leftChannel = splitResult.Value.left;//samples;//
                float[] rightChannel = splitResult.Value.right;

                float normFactor = GetNormFactor(leftChannel);
                leftChannel = leftChannel.Select(x => x * normFactor).ToArray();
                ///////////////////////////
                Utils.STFTArgs args = new Utils.STFTArgs();
                args.win_len = 400;
                args.fft_len = 400;
                args.win_type = "hanning";
                args.win_inc = 100;
                // 对音频进行 STFT 变换
                Complex[,,] stftComplexLeft = AudioProcessing.Stft(leftChannel, args, normalized: false, pad_mode: "constant");
                float[,,] spectrumLeft = ConvertComplexToSTFTFormat(stftComplexLeft);
                var magPhaLeft = GetMagPha(stftSpec: spectrumLeft, compressFactor: 0.3f);
                float[] noisy_pha = magPhaLeft.pha.Cast<float>().ToArray();
                float[] noisy_mag = magPhaLeft.mag.Cast<float>().ToArray();
                var inputMeta = _modelSession.InputMetadata;
                var container = new List<NamedOnnxValue>();
                var inputNames = new List<string>();
                var inputValues = new List<FixedBufferOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "noisy_pha")
                    {
                        int[] dim = new int[] { batchSize, 201, noisy_pha.Length / batchSize / 201 };
                        var tensor = new DenseTensor<float>(noisy_pha, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                    if (name == "noisy_mag")
                    {
                        int[] dim = new int[] { batchSize, 201, noisy_mag.Length / batchSize / 201 };
                        var tensor = new DenseTensor<float>(noisy_mag, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                }
                try
                {
                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                    encoderResults = _modelSession.Run(container);

                    if (encoderResults != null)
                    {
                        var encoderResultsArray = encoderResults.ToArray();
                        Tensor<float> mag = encoderResults[0].AsTensor<float>();//amp_g 
                        Tensor<float> pha = encoderResults[1].AsTensor<float>();//pha_g 
                        var mag_arr = To3DArray(mag);
                        var pha_arr = To3DArray(pha);
                        var com = GetCom(mag_arr, pha_arr, compressFactor: 0.3f);
                        var spec = GetBatch2(com: com, b: 0);//2
                        Complex[,,] spectrumX = ConvertSTFTFormatToComplex2(spec);//2
                        float[] output = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false);
                        output = output.Select(x => x / normFactor).ToArray();
                        ModelOutputEntity modelOutput = new ModelOutputEntity();
                        modelOutput.StemName = "vocals";
                        modelOutput.StemContents = output;
                        modelOutputEntities.Add(modelOutput);
                    }
                }
                catch (Exception ex)
                {
                    //
                }
            }
            return modelOutputEntities;
        }

        public List<ModelOutputEntity> ModelProj_mono(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            Utils.STFTArgs args = new Utils.STFTArgs();
            args.win_len = 4096;
            args.fft_len = 4096;
            args.win_type = "hanning";
            args.win_inc = 1024;
            // 对音频进行 STFT 变换
            Complex[,,] stftComplex = AudioProcessing.Stft(samples, args, normalized: false, pad_mode: "constant");
            float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
            float[,,,] stft = MergeSpectrums(spectrum, spectrum);
            float[,,] mag = ProcessSTFTAndComputeMagnitude(stft, 1024);
            stft = CropSTFTFrequencies(stft, 1024);
            float[] input = stft.Cast<float>().ToArray();
            float[] input_mag = mag.Cast<float>().ToArray();
            var inputMeta = _modelSession.InputMetadata;
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { 2, 1024, input.Length / 2 / 1024 / 2, 2 };
                    var tensor = new DenseTensor<float>(input, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "input_mag")
                {
                    int[] dim = new int[] { 2, 1024, input_mag.Length / 2 / 1024 };
                    var tensor = new DenseTensor<float>(input_mag, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _modelSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    foreach (var encoderResult in encoderResultsArray)
                    {
                        string name = encoderResult.Name;
                        var outputTensor = encoderResult.AsTensor<float>();
                        (Tensor<float> channel0, Tensor<float> channel1) channels = SplitStereoSTFT(outputTensor);
                        var spec = To3DArray(channels.channel0);
                        //Complex[,,] spectrumX1 = ConvertSTFTFormatToComplex2_2(spec);
                        Complex[,,] spectrumX = ConvertSTFTFormatToComplex(spec);
                        float[] output = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false);
                        ModelOutputEntity modelOutput = new ModelOutputEntity();
                        modelOutput.StemName = name;
                        modelOutput.StemContents = output;
                        modelOutputEntities.Add(modelOutput);
                    }

                }
            }
            catch (Exception ex)
            {
                //
            }
            return modelOutputEntities;
        }

        public List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1)
        {
            return null;
        }

        public float[,,] To3DArray(Tensor<float> tensor)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be 3-dimensional");

            var dimensions = tensor.Dimensions;
            float[,,] result = new float[dimensions[0], dimensions[1], dimensions[2]];

            // 通用索引访问
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

        public static Complex[,] ConvertToComplex(float[,,] floatArray)
        {
            // 检查输入数组的维度
            if (floatArray.Rank != 3 || floatArray.GetLength(0) != 1)
            {
                throw new ArgumentException("输入数组必须是三维数组，且第一维长度为1。");
            }

            int rows = floatArray.GetLength(1);
            int cols = floatArray.GetLength(2);

            // 创建目标复数数组
            Complex[,] complexArray = new Complex[rows, cols];

            // 遍历并转换每个元素
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // 从输入数组获取实部，虚部设为0
                    float real = floatArray[0, i, j];
                    complexArray[i, j] = new Complex(real, 0);
                }
            }

            return complexArray;
        }

        /// <summary>
        /// 将两个float[,,]频谱合并为一个float[,,,]（增加通道维度）
        /// </summary>
        /// <param name="spectrum1">第一个频谱，形状为[2049, 427, 2]</param>
        /// <param name="spectrum2">第二个频谱，形状为[2049, 427, 2]</param>
        /// <returns>合并后的四维数组，形状为[2, 2049, 427, 2]</returns>
        public static float[,,,] MergeSpectrums(float[,,] spectrum1, float[,,] spectrum2)
        {
            // 验证输入数组维度
            if (spectrum1.Rank != 3 || spectrum2.Rank != 3)
                throw new ArgumentException("输入数组必须是三维数组");

            // 获取数组维度（假设两个数组维度相同）
            int freqBins = spectrum1.GetLength(0);    // 2049
            int timeFrames = spectrum1.GetLength(1);  // 427
            int complexParts = spectrum1.GetLength(2); // 2

            // 创建新的四维数组 [2, 2049, 427, 2]
            float[,,,] mergedSpectrum = new float[2, freqBins, timeFrames, complexParts];

            // 复制第一个频谱到通道0
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    for (int c = 0; c < complexParts; c++)
                    {
                        mergedSpectrum[0, f, t, c] = spectrum1[f, t, c];
                    }
                }
            }

            // 复制第二个频谱到通道1
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    for (int c = 0; c < complexParts; c++)
                    {
                        mergedSpectrum[1, f, t, c] = spectrum2[f, t, c];
                    }
                }
            }

            return mergedSpectrum;
        }

        /// <summary>
        /// 裁剪STFT频谱的频率范围
        /// 等效于Python中的: stft = stft[:, :self.F, :, :]
        /// </summary>
        /// <param name="stft">输入的STFT频谱，四维数组[通道数, 频率, 时间, 实虚部]</param>
        /// <param name="maxFreq">要保留的最大频率索引（包含）</param>
        /// <returns>裁剪后的STFT频谱，四维数组[通道数, maxFreq, 时间, 实虚部]</returns>
        public static float[,,,] CropSTFTFrequencies(float[,,,] stft, int maxFreq)
        {
            // 获取输入维度
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);
            int complexParts = stft.GetLength(3); // 通常为2（实部和虚部）

            // 确保maxFreq不超过原始频率范围
            if (maxFreq >= originalFreqBins)
                throw new ArgumentException($"maxFreq({maxFreq})必须小于原始频率范围({originalFreqBins})");

            // 创建裁剪后的数组 [通道数, maxFreq, 时间, 实虚部]
            float[,,,] croppedStft = new float[numChannels, maxFreq, timeFrames, complexParts];

            // 复制数据（仅保留前maxFreq个频率点）
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
        /// 处理STFT频谱并计算幅度谱
        /// 等效于Python中的:
        /// stft = stft[:, :self.F, :, :]
        /// real = stft[:, :, :, 0]
        /// im = stft[:, :, :, 1]
        /// mag = torch.sqrt(real ** 2 + im ** 2)
        /// </summary>
        /// <param name="stft">输入的STFT频谱，四维数组[通道数, 频率, 时间, 实虚部]</param>
        /// <param name="maxFreq">要保留的最大频率索引（对应Python中的self.F）</param>
        /// <returns>幅度谱，三维数组[通道数, 频率, 时间]</returns>
        public static float[,,] ProcessSTFTAndComputeMagnitude(float[,,,] stft, int maxFreq)
        {
            // 获取输入维度
            int numChannels = stft.GetLength(0);
            int originalFreqBins = stft.GetLength(1);
            int timeFrames = stft.GetLength(2);

            // 确保maxFreq不超过原始频率范围
            if (maxFreq > originalFreqBins)
                throw new ArgumentException($"maxFreq({maxFreq})超过原始频率范围({originalFreqBins})");

            // 创建输出幅度谱数组 [通道数, maxFreq, 时间]
            float[,,] magnitude = new float[numChannels, maxFreq, timeFrames];

            // 处理STFT并计算幅度谱
            for (int ch = 0; ch < numChannels; ch++)
            {
                for (int f = 0; f < maxFreq; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // 获取实部和虚部
                        float real = stft[ch, f, t, 0];
                        float imag = stft[ch, f, t, 1];

                        // 计算幅度：sqrt(real² + imag²)
                        magnitude[ch, f, t] = (float)Math.Sqrt(real * real + imag * imag);
                    }
                }
            }

            return magnitude;
        }
        /// <summary>
        /// 将四维STFT数组[2,1024,427,2]拆分为两个三维数组[1024,427,2]
        /// </summary>
        /// <param name="stft">输入的四维STFT数组[2,1024,427,2]</param>
        /// <returns>包含两个三维数组的元组</returns>
        public static (float[,,] channel0, float[,,] channel1) SplitStereoSTFT(float[,,,] stft)
        {
            // 验证输入维度
            if (stft.Rank != 4 ||
                stft.GetLength(0) != 2 ||
                stft.GetLength(1) != 1024 ||
                stft.GetLength(3) != 2)
            {
                throw new ArgumentException("输入数组必须是[2,1024,427,2]格式的四维数组");
            }
            int dim = stft.GetLength(2);
            // 创建两个三维数组
            float[,,] channel0 = new float[1024, dim, 2]; // 第一个通道
            float[,,] channel1 = new float[1024, dim, 2]; // 第二个通道

            // 复制数据
            for (int f = 0; f < 1024; f++)
            {
                for (int t = 0; t < dim; t++)
                {
                    for (int c = 0; c < 2; c++)
                    {
                        channel0[f, t, c] = stft[0, f, t, c]; // 复制第一个通道
                        channel1[f, t, c] = stft[1, f, t, c]; // 复制第二个通道
                    }
                }
            }

            return (channel0, channel1);
        }

        /// <summary>
        /// 将四维Tensor<float>[2,1024,427,2]拆分为两个三维Tensor<float>[1024,427,2]
        /// </summary>
        /// <param name="stftTensor">输入的四维Tensor[2,1024,427,2]</param>
        /// <returns>包含两个三维Tensor的元组</returns>
        public static (Tensor<float> channel0, Tensor<float> channel1) SplitStereoSTFT(Tensor<float> stftTensor)
        {
            // 验证输入维度
            if (stftTensor.Dimensions.Length != 4 ||
                stftTensor.Dimensions[0] != 2 ||
                stftTensor.Dimensions[1] != 1024 ||
                stftTensor.Dimensions[3] != 2)
            {
                throw new ArgumentException("输入Tensor必须是[2,1024,427,2]格式的四维张量");
            }

            int dim = stftTensor.Dimensions[2];

            // 创建两个三维Tensor
            var channel0 = new DenseTensor<float>(new int[] { 1024, dim, 2 });
            var channel1 = new DenseTensor<float>(new int[] { 1024, dim, 2 });

            // 复制数据
            for (int f = 0; f < 1024; f++)
            {
                for (int t = 0; t < dim; t++)
                {
                    for (int c = 0; c < 2; c++)
                    {
                        channel0[f, t, c] = stftTensor[0, f, t, c]; // 复制第一个通道
                        channel1[f, t, c] = stftTensor[1, f, t, c]; // 复制第二个通道
                    }
                }
            }

            return (channel0, channel1);
        }



        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_modelSession != null)
                    {
                        _modelSession = null;
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~SepProjOfZipEnhancerSe()
        {
            Dispose(_disposed);
        }
    }
}
