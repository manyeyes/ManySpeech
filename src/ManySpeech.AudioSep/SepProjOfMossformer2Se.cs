using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfMossformer2Se : ISepProj,IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 60;
        private int _sampleRate = 48000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public SepProjOfMossformer2Se(SepModel sepModel)
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

        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList=null, int offset=0)
        {
            int batchSize = modelInputs.Count;
            //Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 60).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            samples = samples.Select(x => x * 32768f).ToArray();
            //samples = samples.Select(x => x == 0 ? float.E/100 * 32768f : x).ToArray();
            FrontendConfEntity frontendConfEntity = new FrontendConfEntity();
            WavFrontend wavFrontend = new WavFrontend(frontendConfEntity);
            float[] features = wavFrontend.GetFbank(samples);
            DenseTensor<float> featuresTensor = ConvertToTensor(features, (int)(features.Length/60),60);
            (DenseTensor<float>, DenseTensor<float>) deltas = ProcessFilterBanks(featuresTensor);
            List<DenseTensor<float>> tensorList = new List<DenseTensor<float>>();
            tensorList.Add(featuresTensor);
            tensorList.Add(deltas.Item1);
            tensorList.Add(deltas.Item2);
            DenseTensor<float> concatenateTensors = ConcatenateTensors(tensorList.ToArray(), dim:1);
            features = concatenateTensors.ToArray();
            var inputMeta = _modelSession.InputMetadata;
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "inputs")
                {
                    int[] dim = new int[] { batchSize, features.Length/batchSize/180,180 };                    
                    var tensor = new DenseTensor<float>(features, dim, false);
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
                    var outputTensor = encoderResultsArray[0].AsTensor<float>();

                    Utils.STFTArgs args = new Utils.STFTArgs();
                    args.win_len = 1920;
                    args.fft_len = 1920;
                    args.win_type = "hamming";
                    args.win_inc = 384;

                    float[,,] outList = AudioProcessing.TensorTo3DArray(outputTensor);
                    float[,,] predMask= outList;
                    //predMask= outList[outList.GetLength(0) - 1]; // 获取最后一个输出作为掩码

                    // 执行STFT
                    //Complex[,] stftResult = AudioProcessing.Stft(outputTensor.ToArray(), args);
                    // 对音频进行 STFT 变换
                    Complex[,,] stftComplex = AudioProcessing.Stft(samples, args);
                    float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
                    // 调整掩码维度顺序以匹配频谱
                    // 原始维度顺序: [batch, freq, time] -> 调整为: [time, freq, ?]
                    float[,,] permutedMask = AudioProcessing.PermuteDimensions(predMask, 2, 1, 0);
                    // 将掩码应用到频谱上
                    float[,,] maskedSpec = AudioProcessing.ApplyMask(spectrum, permutedMask);
                    Complex[,] maskedSpecComplex = AudioProcessing.ConvertToComplex(maskedSpec);
                    // 从掩码后的频谱重建音频
                    float[] output0 = AudioProcessing.Istft(maskedSpecComplex, args, samples.Length);

                    // 去掉额外增加的尾部采样
                    int sampleRate = modelInputs[0].SampleRate;
                    float[] output = new float[(int)(samples.Length - sampleRate * 0.5f) / 1];
                    output[output.Length - 1] = 0;
                    Array.Copy(output0, 0, output, 0, output.Length);

                    ModelOutputEntity modelOutput = new ModelOutputEntity();
                    modelOutput.StemName = "vocals";
                    modelOutput.StemContents = output.Select(x=>x/32768f).ToArray();
                    modelOutputEntities.Add(modelOutput);
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
        // 将一维float数组转换为指定形状的二维DenseTensor
        public static DenseTensor<float> ConvertToTensor(float[] data, int rows, int cols)
        {
            if (data.Length != rows * cols)
            {
                throw new ArgumentException($"输入数组长度 ({data.Length}) 不等于指定形状的元素总数 ({rows * cols})");
            }

            // 创建目标形状的张量
            var tensor = new DenseTensor<float>(new int[] { rows, cols });

            // 按行优先方式填充数据
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // 计算一维数组中的索引
                    int linearIndex = i * cols + j;
                    tensor[i, j] = data[linearIndex];
                }
            }

            return tensor;
        }

        /// <summary>
        /// 计算音频特征的delta系数，完全模拟torchaudio.functional.compute_deltas
        /// </summary>
        /// <param name="features">输入特征张量，维度为 [..., freq, time]</param>
        /// <param name="win_length">delta计算窗口长度，必须为奇数</param>
        /// <param name="mode">边界填充模式，支持'replicate'和'zero'</param>
        /// <returns>delta特征张量</returns>
        public static DenseTensor<float> ComputeDeltas(DenseTensor<float> features, int win_length = 5, string mode = "replicate")
        {
            if (features == null)
                throw new ArgumentNullException(nameof(features), "特征矩阵不能为null");

            int freq = features.Dimensions[0];
            int time = features.Dimensions[1];

            if (freq == 0 || time == 0)
                throw new ArgumentException("特征矩阵不能为空", nameof(features));

            if (win_length % 2 == 0)
                throw new ArgumentException("窗口长度必须为奇数", nameof(win_length));

            if (win_length < 3)
                throw new ArgumentException("窗口长度至少为3", nameof(win_length));

            if (mode != "replicate" && mode != "zero")
                throw new ArgumentException("不支持的填充模式，支持'replicate'和'zero'", nameof(mode));

            int N = (win_length - 1) / 2;
            DenseTensor<float> deltas = new DenseTensor<float>(new int[] { freq, time });

            // 计算分母：2 * sum(n^2) for n=1 to N
            float denominator = 0;
            for (int n = 1; n <= N; n++)
            {
                denominator += n * n;
            }
            denominator *= 2;

            // 对每个频率维度计算delta
            for (int f = 0; f < freq; f++)
            {
                for (int t = 0; t < time; t++)
                {
                    float numerator = 0;

                    // 计算分子：sum(n * (c[t+n] - c[t-n])) for n=1 to N
                    for (int n = 1; n <= N; n++)
                    {
                        float forward = GetValueWithPadding(features, f, t + n, time, mode);
                        float backward = GetValueWithPadding(features, f, t - n, time, mode);
                        numerator += n * (forward - backward);
                    }

                    deltas[f, t] = numerator / denominator;
                }
            }

            return deltas;
        }

        /// <summary>
        /// 根据指定的填充模式获取值
        /// </summary>
        private static float GetValueWithPadding(DenseTensor<float> features, int freq, int timeIdx, int maxTime, string mode)
        {
            if (timeIdx < 0)
            {
                return mode == "replicate" ? features[freq, 0] : 0;
            }
            else if (timeIdx >= maxTime)
            {
                return mode == "replicate" ? features[freq, maxTime - 1] : 0;
            }
            else
            {
                return features[freq, timeIdx];
            }
        }

        // 处理滤波器组并计算deltas和delta-deltas
        public static (DenseTensor<float>, DenseTensor<float>) ProcessFilterBanks(DenseTensor<float> fbanks)
        {
            // 转置：(frames, features) -> (features, frames)
            var fbanksTr = TransposeToDenseTensor(fbanks);

            // 计算一阶差分
            var fbankDelta = ComputeDeltas(fbanksTr);

            // 计算二阶差分
            var fbankDeltaDelta = ComputeDeltas(fbankDelta);

            // 转回原始维度顺序
            var fbankDeltaTr = TransposeToDenseTensor(fbankDelta);
            var fbankDeltaDeltaTr = TransposeToDenseTensor(fbankDeltaDelta);

            return (fbankDeltaTr, fbankDeltaDeltaTr);
        }

        
        // 对二维数组进行转置并返回DenseTensor<float>
        public static DenseTensor<float> TransposeToDenseTensor(DenseTensor<float> source)
        {
            int rows = source.Dimensions[0];
            int cols = source.Dimensions[1];

            // 创建转置后的数组
            float[,] transposedArray = new float[cols, rows];

            // 填充转置后的数据
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposedArray[j, i] = source[i, j];
                }
            }

            // 将二维数组转换为一维数组(按行优先)
            float[] flatArray = new float[cols * rows];
            int index = 0;
            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    flatArray[index++] = transposedArray[i, j];
                }
            }

            // 创建DenseTensor并指定形状
            return new DenseTensor<float>(flatArray, new int[] { cols, rows });
        }
        // 对二维浮点数组进行轴交换(转置)
        public static float[,] Transpose(float[,] source)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);

            // 创建目标矩阵并交换维度
            float[,] result = new float[cols, rows];

            // 填充转置后的数据
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = source[i, j];
                }
            }

            return result;
        }
        // 张量转置辅助方法
        private static DenseTensor<float> TransposeTensor(DenseTensor<float> tensor)
        {
            int rows = tensor.Dimensions[0];
            int cols = tensor.Dimensions[1];

            var result = new DenseTensor<float>(new int[] { cols, rows });

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = tensor[i, j];
                }
            }

            return result;
        }

        // 沿指定维度拼接多个张量
        public static DenseTensor<float> ConcatenateTensors(DenseTensor<float>[] tensors, int dim)
        {
            if (tensors == null || tensors.Length == 0)
                throw new ArgumentException("至少需要一个张量进行拼接");

            // 验证所有张量的维度相同
            int rank = tensors[0].Rank;
            if (dim < 0 || dim >= rank)
                throw new ArgumentException($"维度索引 {dim} 超出范围（张量维度为 {rank}）");

            // 验证所有张量的非拼接维度相同
            var referenceShape = tensors[0].Dimensions.ToArray();
            for (int i = 1; i < tensors.Length; i++)
            {
                if (tensors[i].Rank != rank)
                    throw new ArgumentException($"张量 {i} 的维度数 ({tensors[i].Rank}) 与第一个张量 ({rank}) 不匹配");

                for (int d = 0; d < rank; d++)
                {
                    if (d != dim && tensors[i].Dimensions[d] != referenceShape[d])
                        throw new ArgumentException($"张量 {i} 在维度 {d} 上的大小 ({tensors[i].Dimensions[d]}) 与第一个张量 ({referenceShape[d]}) 不匹配");
                }
            }

            // 计算输出张量的形状
            int[] outputShape = new int[rank];
            outputShape[0] = referenceShape[0];
            //Array.Copy(referenceShape, outputShape, rank);
            foreach (var tensor in tensors)
            {
                outputShape[dim] += tensor.Dimensions[dim];
            }

            // 创建输出张量
            var output = new DenseTensor<float>(outputShape);

            // 计算每个张量在拼接维度上的偏移量
            int[] offsets = new int[tensors.Length];
            int currentOffset = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                offsets[i] = currentOffset;
                currentOffset += tensors[i].Dimensions[dim];
            }

            // 对每个张量进行处理
            for (int tensorIndex = 0; tensorIndex < tensors.Length; tensorIndex++)
            {
                var tensor = tensors[tensorIndex];
                int offset = offsets[tensorIndex];

                // 使用多维索引遍历源张量的每个元素
                int[] indices = new int[rank];
                TraverseTensor(tensor, indices, 0, (srcIndices) =>
                {
                    // 复制除拼接维度外的所有索引
                    int[] targetIndices = new int[rank];
                    Array.Copy(srcIndices, targetIndices, rank);

                    // 修改拼接维度的索引，加上当前张量的偏移量
                    targetIndices[dim] = srcIndices[dim] + offset;

                    // 获取源值和目标位置
                    float value = tensor[srcIndices];
                    output[targetIndices] = value;
                });
            }

            return output;
        }

        // 递归遍历张量的每个元素
        private static void TraverseTensor(DenseTensor<float> tensor, int[] indices, int currentDim, Action<int[]> action)
        {
            if (currentDim == tensor.Rank)
            {
                action(indices);
                return;
            }

            int dimSize = tensor.Dimensions[currentDim];
            for (int i = 0; i < dimSize; i++)
            {
                indices[currentDim] = i;
                TraverseTensor(tensor, indices, currentDim + 1, action);
            }
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
        ~SepProjOfMossformer2Se()
        {
            Dispose(_disposed);
        }
    }
}
