using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfGtcrnSe : ISepProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public SepProjOfGtcrnSe(SepModel sepModel)
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
        /// Converts back from float[257,609,2] STFT format to Complex[257,1,609]
        /// </summary>
        /// <param name="stftFormat">STFT format array with shape [1, 2*freq_bins, time_frames]</param>
        /// <returns>Complex spectrogram with shape [freq_bins, 1, time_frames]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int freqBins = stftFormat.GetLength(0);     // 频率 bins (257)
            int timeFrames = stftFormat.GetLength(1);   // 时间帧 (609)

            // 创建目标复数数组，中间维度为1
            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            // 遍历每个频率和时间点
            for (int f = 0; f < freqBins; f++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // 从最后一维提取实部和虚部
                    float real = stftFormat[f, t, 0];    // 实部
                    float imag = stftFormat[f, t, 1];    // 虚部

                    // 创建复数，注意中间维度索引为0
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }
            }

            return complexSpectrogram;
        }

        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            //samples = samples.Select(x => x / 32768f).ToArray();
            Utils.STFTArgs args = new Utils.STFTArgs();
            args.win_len = 512;
            args.fft_len = 512;
            args.win_type = "hanning";
            args.win_inc = 256;
            // 对音频进行 STFT 变换
            Complex[,,] stftComplex = AudioProcessing.Stft(samples, args, normalized: false);
            float[,,] spectrum = ConvertComplexToSTFTFormat(stftComplex);
            float[] padSequence = spectrum.Cast<float>().ToArray();
            var inputMeta = _modelSession.InputMetadata;
            List<ModelOutputEntity> modelOutputEntities = new List<ModelOutputEntity>();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { batchSize, 257, padSequence.Length / batchSize / 257 / 2, 2 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
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

                    var spec = To3DArray(outputTensor);
                    Complex[,,] spectrumX = ConvertSTFTFormatToComplex(spec);
                    float[] output0 = AudioProcessing.Istft(spectrumX, args, samples.Length, normalized: false);

                    // 去掉额外增加的尾部采样
                    int sampleRate = modelInputs[0].SampleRate;
                    int channels = modelInputs[0].Channels;
                    float[] output = new float[(int)(samples.Length - sampleRate * channels * 0.1f) / channels];
                    Array.Copy(output0, 0, output, 0, output.Length);

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

        public List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1)
        {
            return null;
        }

        public float[,,] To3DArray(Tensor<float> tensor)
        {
            int[] dimensions = tensor.Dimensions.ToArray();

            // 检查是否为3维或4维
            if (tensor.Rank != 3 && tensor.Rank != 4)
                throw new ArgumentException("Tensor must be 3-dimensional or 4-dimensional");

            // 处理4维Tensor（假设第一维为1）
            if (tensor.Rank == 4)
            {
                if (dimensions[0] != 1)
                    throw new ArgumentException("4-dimensional tensor must have first dimension equal to 1");

                dimensions = new int[] { dimensions[1], dimensions[2], dimensions[3] };
            }

            float[,,] result = new float[dimensions[0], dimensions[1], dimensions[2]];

            // 通用索引访问
            var indices = new int[tensor.Rank];
            for (indices[0] = 0; indices[0] < dimensions[0]; indices[0]++)
            {
                for (indices[1] = 0; indices[1] < dimensions[1]; indices[1]++)
                {
                    for (indices[2] = 0; indices[2] < dimensions[2]; indices[2]++)
                    {
                        // 对于4维Tensor，固定第一维为0
                        result[indices[0], indices[1], indices[2]] =
                            tensor.Rank == 3
                                ? tensor[indices[0], indices[1], indices[2]]
                                : tensor[0, indices[0], indices[1], indices[2]];
                    }
                }
            }

            return result;
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
        ~SepProjOfGtcrnSe()
        {
            Dispose(_disposed);
        }
    }
}
