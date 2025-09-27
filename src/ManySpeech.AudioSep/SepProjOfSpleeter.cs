using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfSpleeter : ISepProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public SepProjOfSpleeter(SepModel sepModel)
        {
            _modelSession = sepModel.ModelSession;
            _customMetadata = sepModel.CustomMetadata;
            _featureDim = sepModel.FeatureDim;
            _sampleRate = sepModel.SampleRate;
            _channels = sepModel.Channels;
            _chunkLength = sepModel.ChunkLength;
            _shiftLength = sepModel.ShiftLength;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Channels { get => _channels; set => _channels = value; }

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

        /// <summary>
        /// 将 float[1024, 427, 2] 格式的 STFT 转换为 Complex[2049, 1, 427] 格式的复数频谱
        /// </summary>
        /// <param name="stftFormat">输入的 STFT 数据，格式为 [频率, 时间, 实虚部]</param>
        /// <returns>复数频谱，格式为 [频率, 1, 时间]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // 原始频率点数 (1024)
            int timeFrames = stftFormat.GetLength(1);   // 时间帧数 (427)
            int fullFreqBins = 2049;                    // 完整频率点数 (2049)

            // 验证输入维度
            if (freqBins != 1024 || stftFormat.GetLength(2) != 2)
            {
                throw new ArgumentException("输入数组必须是 [1024, 时间帧数, 2] 格式");
            }

            // 创建目标复数数组 [2049, 1, 427]
            Complex[,,] complexSpectrogram = new Complex[fullFreqBins, 1, timeFrames];

            // 遍历每个时间帧
            for (int t = 0; t < timeFrames; t++)
            {
                // 1. 复制正频率部分（0-1023）
                for (int f = 0; f < 1024; f++)
                {
                    float real = stftFormat[f, t, 0];    // 实部
                    float imag = stftFormat[f, t, 1];    // 虚部
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }

                // 2. 生成负频率部分（1025-2048），使用共轭对称性
                for (int f = 1025; f < fullFreqBins; f++)
                {
                    // 对应正频率索引（共轭对称）
                    int posFreqIndex = 2048 - (f - 1024);

                    // 获取正频率部分的复数
                    Complex posFreqValue = complexSpectrogram[posFreqIndex, 0, t];

                    // 计算共轭值（实部相同，虚部取反）
                    Complex conjValue = Complex.Conjugate(posFreqValue);
                    complexSpectrogram[f, 0, t] = conjValue;
                }

                // 3. 特殊处理 Nyquist 频率点（索引 1024）
                // 对于实数信号，Nyquist 频率点的虚部应为 0
                complexSpectrogram[1024, 0, t] = new Complex(stftFormat[1023, t, 0], 0);
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

        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            /////////////////////////////
            var splitResult = SplitStereoToMono(samples);
            if (splitResult.HasValue)
            {
                float[] leftChannel = splitResult.Value.left;
                float[] rightChannel = splitResult.Value.right;

                ///////////////////////////
                Utils.STFTArgs args = new Utils.STFTArgs();
                args.win_len = 4096;
                args.fft_len = 4096;
                args.win_type = "hanning";
                args.win_inc = 1024;
                // 对音频进行 STFT 变换
                Complex[,,] stftComplexLeft = AudioProcessing.Stft(leftChannel, args, normalized: false, pad_mode: "constant");
                float[,,] spectrumLeft = ConvertComplexToSTFTFormat(stftComplexLeft);
                Complex[,,] stftComplexRight = AudioProcessing.Stft(rightChannel, args, normalized: false, pad_mode: "constant");
                float[,,] spectrumRight = ConvertComplexToSTFTFormat(stftComplexRight);
                float[,,,] stft = MergeSpectrums(spectrumLeft, spectrumRight);
                float[,,] mag = ProcessSTFTAndComputeMagnitude(stft, 1024);
                stft = CropSTFTFrequencies(stft, 1024);
                float[] input = stft.Cast<float>().ToArray();
                float[] input_mag = mag.Cast<float>().ToArray();
                var inputMeta = _modelSession.InputMetadata;
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

                            //var spec = To3DArray(outputTensor);
                            int sampleRate = modelInputs[0].SampleRate;
                            (Tensor<float> channel0, Tensor<float> channel1) channels = SplitStereoSTFT(outputTensor);
                            var spec0 = To3DArray(channels.channel0);
                            Complex[,,] spectrumX0 = ConvertSTFTFormatToComplex(spec0);
                            float[] output0 = AudioProcessing.Istft(spectrumX0, args, samples.Length, normalized: false);
                            float[] left=new float[(int)(samples.Length- sampleRate * 0.5f) /2];
                            Array.Copy(output0,0, left, 0, left.Length);
                            var spec1 = To3DArray(channels.channel1);
                            Complex[,,] spectrumX1 = ConvertSTFTFormatToComplex(spec1);
                            float[] output1 = AudioProcessing.Istft(spectrumX1, args, samples.Length, normalized: false);
                            float[] right = new float[(int)(samples.Length - sampleRate * 0.5f) / 2];
                            Array.Copy(output1, 0, right, 0, right.Length);
                            float[] output= MergeMonoToStereo(left, right);
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
        ~SepProjOfSpleeter()
        {
            Dispose(_disposed);
        }
    }
}
