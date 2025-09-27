using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfUvr : ISepProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private const int F = 2048;
        private const int T = 512;
        public SepProjOfUvr(SepModel sepModel)
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
        /// 将 float[F, T, 2] 格式的 STFT 转换为 Complex[F, 1, T] 格式的复数频谱
        /// </summary>
        /// <param name="stftFormat">输入的 STFT 数据，格式为 [频率, 时间, 实虚部]</param>
        /// <returns>复数频谱，格式为 [频率, 1, 时间]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // 频率点数 (F)
            int timeFrames = stftFormat.GetLength(1);   // 时间帧数 (T)

            // 验证输入维度
            if (freqBins != F || stftFormat.GetLength(2) != 2)
            {
                throw new ArgumentException("输入数组必须是 [F, 时间帧数, 2] 格式");
            }

            // 创建目标复数数组 [F, 1, T]
            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            // 遍历每个频率点和时间帧
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 获取实部和虚部
                    float real = stftFormat[f, t, 0];
                    float imag = stftFormat[f, t, 1];

                    // 创建复数并存储到输出数组中
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }
            }

            return complexSpectrogram;
        }

        /// <summary>
        /// 将 float[F, T, 2] 格式的 STFT 转换为 Complex[F, 2, T] 格式的复数频谱
        /// </summary>
        /// <param name="stftFormat">输入的 STFT 数据，格式为 [频率, 时间, 实虚部]</param>
        /// <returns>复数频谱，格式为 [频率, 2通道, 时间]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex2(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // 频率点数 (F)
            int timeFrames = stftFormat.GetLength(1);   // 时间帧数 (T)

            // 验证输入维度
            //if (freqBins != F || stftFormat.GetLength(2) != 2)
            //{
            //    throw new ArgumentException("输入数组必须是 [F, 时间帧数, 2] 格式");
            //}

            // 创建目标复数数组 [F, 2, T]
            Complex[,,] complexSpectrogram = new Complex[freqBins, 2, timeFrames];

            // 遍历每个频率点和时间帧
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 获取实部和虚部
                    float real = stftFormat[f, t, 0];
                    float imag = stftFormat[f, t, 1];

                    // 存储到复数数组中，第二维对应通道
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                    complexSpectrogram[f, 1, t] = new Complex(real, imag); // 复制到第二个通道
                }
            }

            return complexSpectrogram;
        }

        /// <summary>
        /// 将float32[batch_size,4,F,T]格式的数据转换为复数频谱表示，按批次返回List<float[,,]>
        /// 其中每个float[,,]的形状为[F, T, 2]，对应[频率, 时间, 实虚部]
        /// </summary>
        /// <param name="input">输入数据，格式为[batch_size, 4, F, T]</param>
        /// <returns>复数频谱列表，每个元素格式为[F, T, 2]</returns>
        public static List<float[,,]> ConvertBatchSpectrums(float[,,,] input)
        {
            int batchSize = input.GetLength(0);
            int freqBins = input.GetLength(2);    // F
            int timeFrames = input.GetLength(3);  // T

            // 验证输入维度
            if (input.GetLength(1) != 4)
            {
                throw new ArgumentException("输入数组的第2个维度必须是4，表示两个通道的实部和虚部");
            }

            // 初始化结果列表
            List<float[,,]> result = new List<float[,,]>(batchSize);

            // 处理每个批次样本
            for (int b = 0; b < batchSize; b++)
            {
                // 创建新的三维数组 [F, T, 2]
                float[,,] complexSpectrum = new float[freqBins, timeFrames, 2];

                // 遍历每个频率和时间点
                for (int f = 0; f < freqBins; f++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        // 从输入的4个通道中提取数据
                        // 假设通道顺序为：左声道实部、左声道虚部、右声道实部、右声道虚部
                        float leftReal = input[b, 0, f, t];   // 左声道实部
                        float leftImag = input[b, 1, f, t];   // 左声道虚部
                        float rightReal = input[b, 2, f, t];  // 右声道实部
                        float rightImag = input[b, 3, f, t];  // 右声道虚部

                        // 合并左右声道（取平均值）
                        float realPart = (leftReal + rightReal) / 2.0f;
                        float imagPart = (leftImag + rightImag) / 2.0f;

                        // 存入结果数组 [频率, 时间, 实虚部]
                        complexSpectrum[f, t, 0] = realPart;   // 实部
                        complexSpectrum[f, t, 1] = imagPart;   // 虚部
                    }
                }

                // 添加到结果列表
                result.Add(complexSpectrum);
            }

            return result;
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
            //int F = 2048;// F; 
            int batchSize = modelInputs.Count;//535374-441000
            int chunkSize = ((T - 1) * 1024 + F * 2 - 1) * 2;
            int tailLen = chunkSize - modelInputs[0].Speech.Length;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs, tailLen: tailLen);
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            /////////////////////////////
            var splitResult = SplitStereoToMono(samples);
            if (splitResult.HasValue)
            {
                float[] leftChannel = splitResult.Value.left;
                float[] rightChannel = splitResult.Value.right;

                ///////////////////////////
                Utils.STFTArgs args = new Utils.STFTArgs();
                args.win_len = F * 2 - 1;
                args.fft_len = F * 2 - 1;
                args.win_type = "hanning";
                args.win_inc = 1024;
                try
                {
                    // 对音频进行 STFT 变换
                    Complex[,,] stftComplexLeft = AudioProcessing.Stft(leftChannel, args, normalized: false);
                    float[,,] spectrumLeft = ConvertComplexToSTFTFormat(stftComplexLeft);
                    Complex[,,] stftComplexRight = AudioProcessing.Stft(rightChannel, args, normalized: false);
                    float[,,] spectrumRight = ConvertComplexToSTFTFormat(stftComplexRight);
                    float[,,] stft = MergeSpectrums(spectrumLeft, spectrumRight);
                    //float[,,] mag = ProcessSTFTAndComputeMagnitude(stft, 1024);
                    //stft = CropSTFTFrequencies(stft, 1024);
                    float[] input = stft.Cast<float>().ToArray();
                    //float[] input_mag = mag.Cast<float>().ToArray();
                    var inputMeta = _modelSession.InputMetadata;
                    var container = new List<NamedOnnxValue>();
                    var inputNames = new List<string>();
                    var inputValues = new List<FixedBufferOnnxValue>();
                    foreach (var name in inputMeta.Keys)
                    {
                        if (name == "input")
                        {
                            int[] dim = new int[] { batchSize, 4, F, T };
                            var tensor = new DenseTensor<float>(input, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }
                    }

                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                    encoderResults = _modelSession.Run(container);

                    if (encoderResults != null)
                    {
                        var encoderResultsArray = encoderResults.ToArray();
                        foreach (var encoderResult in encoderResultsArray)
                        {
                            string name = encoderResult.Name;
                            var outputTensor = encoderResult.AsTensor<float>();

                            var tensorList = ConvertTensorToList(outputTensor);
                            //Complex[,,] spectrumX = ConvertSTFTFormatToComplex(tensorList[0]);
                            //float[] output = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false);
                            int sampleRate = modelInputs[0].SampleRate;
                            (Tensor<float> channel0, Tensor<float> channel1) channels = SplitStereoSTFT(tensorList[0]);
                            var spec0 = To3DArray(channels.channel0);
                            Complex[,,] spectrumX0 = ConvertSTFTFormatToComplex(spec0);
                            float[] output0 = AudioProcessing.Istft(spectrumX0, args, samples.Length, normalized: false);
                            float[] left = new float[(int)(samples.Length - tailLen - sampleRate * 0.5f) / 2];
                            Array.Copy(output0, 0, left, 0, left.Length);
                            var spec1 = To3DArray(channels.channel1);
                            Complex[,,] spectrumX1 = ConvertSTFTFormatToComplex(spec1);
                            float[] output1 = AudioProcessing.Istft(spectrumX1, args, samples.Length, normalized: false);
                            float[] right = new float[(int)(samples.Length - tailLen - sampleRate * 0.5f) / 2];
                            Array.Copy(output1, 0, right, 0, right.Length);
                            float[] output = MergeMonoToStereo(left, right);
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
            float[,,] stft = MergeSpectrums(spectrum, spectrum);
            //float[,,] mag = ProcessSTFTAndComputeMagnitude(stft, 1024);
            //stft = CropSTFTFrequencies(stft, 1024);
            float[] input = stft.Cast<float>().ToArray();
            //float[] input_mag = mag.Cast<float>().ToArray();
            var inputMeta = _modelSession.InputMetadata;
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { 1, 4, F, T };
                    var tensor = new DenseTensor<float>(input, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                //if (name == "input_mag")
                //{
                //    int[] dim = new int[] { 2, 1024, input_mag.Length / 2 / 1024 };
                //    var tensor = new DenseTensor<float>(input_mag, dim, false);
                //    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                //}
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

        /// <summary>
        /// 将float32[batch_size,4,F,T]格式的Tensor转换为List<Tensor<float>>
        /// 其中每个Tensor的形状为[F, T, 4]，对应[频率, 时间, 通道]
        /// </summary>
        /// <param name="tensor">输入的Tensor，格式为[batch_size, 4, F, T]</param>
        /// <returns>复数频谱列表，每个元素格式为[F, T, 4]的Tensor</returns>
        public List<Tensor<float>> ConvertTensorToList(Tensor<float> tensor)
        {
            if (tensor.Rank != 4)
                throw new ArgumentException("Tensor must be 4-dimensional with shape [batch_size, 4, F, T]");

            int batchSize = tensor.Dimensions[0];
            int channels = tensor.Dimensions[1];     // 4
            int freqBins = tensor.Dimensions[2];     // F
            int timeFrames = tensor.Dimensions[3];   // T

            // 初始化结果列表
            List<Tensor<float>> result = new List<Tensor<float>>(batchSize);

            // 处理每个批次样本
            for (int b = 0; b < batchSize; b++)
            {
                // 创建新的三维Tensor [F, T, 4]
                var sampleTensor = new DenseTensor<float>(new int[] { freqBins, timeFrames, channels });

                // 通用索引访问
                var indices = new int[4];
                indices[0] = b; // 设置批次索引

                for (int f = 0; f < freqBins; f++)
                {
                    indices[2] = f; // 设置频率索引

                    for (int t = 0; t < timeFrames; t++)
                    {
                        indices[3] = t; // 设置时间索引

                        for (int c = 0; c < channels; c++)
                        {
                            indices[1] = c; // 设置通道索引

                            // 从输入Tensor中获取值并赋值到新的Tensor
                            sampleTensor[f, t, c] = tensor[indices];
                        }
                    }
                }

                // 添加到结果列表
                result.Add(sampleTensor);
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
        /// 将两个float[,,]频谱合并为一个float[,,]（交错实部和虚部）
        /// </summary>
        /// <param name="spectrum1">第一个频谱，形状为[F, T, 2]</param>
        /// <param name="spectrum2">第二个频谱，形状为[F, T, 2]</param>
        /// <returns>合并后的三维数组，形状为[4, F, T]</returns>
        public static float[,,] MergeSpectrums(float[,,] spectrum1, float[,,] spectrum2)
        {
            // 验证输入数组维度
            if (spectrum1.Rank != 3 || spectrum2.Rank != 3)
                throw new ArgumentException("输入数组必须是三维数组");

            // 获取数组维度（假设两个数组维度相同）
            int freqBins = spectrum1.GetLength(0);    // F
            int timeFrames = spectrum1.GetLength(1);  // T
            int complexParts = spectrum1.GetLength(2); // 2

            // 验证输入数组维度是否匹配
            if (spectrum2.GetLength(0) != freqBins ||
                spectrum2.GetLength(1) != timeFrames ||
                spectrum2.GetLength(2) != complexParts)
            {
                throw new ArgumentException("两个输入数组的维度必须匹配");
            }

            // 创建新的三维数组 [4, F, T]
            float[,,] mergedSpectrum = new float[4, freqBins, timeFrames];

            // 复制第一个频谱的实部到通道0
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[0, f, t] = spectrum1[f, t, 0]; // 实部
                }
            }

            // 复制第一个频谱的虚部到通道1
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[1, f, t] = spectrum1[f, t, 1]; // 虚部
                }
            }

            // 复制第二个频谱的实部到通道2
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[2, f, t] = spectrum2[f, t, 0]; // 实部
                }
            }

            // 复制第二个频谱的虚部到通道3
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    mergedSpectrum[3, f, t] = spectrum2[f, t, 1]; // 虚部
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
        /// 将三维数组float[F,T,4]拆分为两个三维数组float[F,T,2]
        /// 对应左声道和右声道的复数频谱（实部和虚部）
        /// </summary>
        /// <param name="inputArray">输入的三维数组[F,T,4]</param>
        /// <returns>包含两个三维数组的元组，分别表示左声道和右声道</returns>
        public static (float[,,] channel0, float[,,] channel1) SplitStereoSTFT(float[,,] inputArray)
        {
            // 验证输入维度
            if (inputArray.Rank != 3 ||
                inputArray.GetLength(0) != F ||
                inputArray.GetLength(1) != T ||
                inputArray.GetLength(2) != 4)
            {
                throw new ArgumentException("输入数组必须是[F,T,4]格式的三维数组");
            }

            // 创建两个三维数组，每个表示一个声道的复数频谱
            float[,,] leftChannel = new float[F, T, 2];
            float[,,] rightChannel = new float[F, T, 2];

            // 复制数据
            for (int f = 0; f < F; f++)
            {
                for (int t = 0; t < T; t++)
                {
                    // 从输入数组的第0和第1通道复制到左声道数组的第0和第1通道（实部和虚部）
                    leftChannel[f, t, 0] = inputArray[f, t, 0];  // 左声道实部
                    leftChannel[f, t, 1] = inputArray[f, t, 1];  // 左声道虚部

                    // 从输入数组的第2和第3通道复制到右声道数组的第0和第1通道（实部和虚部）
                    rightChannel[f, t, 0] = inputArray[f, t, 2];  // 右声道实部
                    rightChannel[f, t, 1] = inputArray[f, t, 3];  // 右声道虚部
                }
            }

            return (leftChannel, rightChannel);
        }

        /// <summary>
        /// 将三维Tensor<float>[F,T,4]拆分为两个三维Tensor<float>[F,T,2]
        /// 对应左声道和右声道的复数频谱（实部和虚部）
        /// </summary>
        /// <param name="inputTensor">输入的三维Tensor[F,T,4]</param>
        /// <returns>包含两个三维Tensor的元组，分别表示左声道和右声道</returns>
        public static (Tensor<float> leftChannel, Tensor<float> rightChannel) SplitStereoSTFT(Tensor<float> inputTensor)
        {
            // 验证输入维度
            if (inputTensor.Dimensions.Length != 3 ||
                inputTensor.Dimensions[0] != F ||
                inputTensor.Dimensions[1] != T ||
                inputTensor.Dimensions[2] != 4)
            {
                throw new ArgumentException("输入Tensor必须是[F,T,4]格式的三维张量");
            }

            // 创建两个三维Tensor，每个表示一个声道的复数频谱
            var leftChannel = new DenseTensor<float>(new int[] { F, T, 2 });
            var rightChannel = new DenseTensor<float>(new int[] { F, T, 2 });

            // 复制数据
            for (int f = 0; f < F; f++)
            {
                for (int t = 0; t < T; t++)
                {
                    // 从输入Tensor的第0和第1通道复制到左声道Tensor的第0和第1通道（实部和虚部）
                    leftChannel[f, t, 0] = inputTensor[f, t, 0];  // 左声道实部
                    leftChannel[f, t, 1] = inputTensor[f, t, 1];  // 左声道虚部

                    // 从输入Tensor的第2和第3通道复制到右声道Tensor的第0和第1通道（实部和虚部）
                    rightChannel[f, t, 0] = inputTensor[f, t, 2];  // 右声道实部
                    rightChannel[f, t, 1] = inputTensor[f, t, 3];  // 右声道虚部
                }
            }

            return (leftChannel, rightChannel);
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
        ~SepProjOfUvr()
        {
            Dispose(_disposed);
        }
    }
}
