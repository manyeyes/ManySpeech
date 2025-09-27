using ManySpeech.AudioSep.Model;
using ManySpeech.AudioSep.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
using System.Numerics;

namespace ManySpeech.AudioSep
{
    internal class SepProjOfFrcrnSe2 : ISepProj, IDisposable
    {
        private bool _disposed;
        private InferenceSession _modelSession;
        private CustomMetadata _customMetadata;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public SepProjOfFrcrnSe2(SepModel sepModel)
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
        /// Converts a Complex[321, 1, 723] array to float[1, 642, 723] STFT format
        /// </summary>
        /// <param name="complexSpectrogram">Input complex spectrogram with shape [freq_bins, 1, time_frames]</param>
        /// <returns>STFT format array with shape [1, 2*freq_bins, time_frames]</returns>
        public static float[,,] ConvertComplexToSTFTFormat(Complex[,,] complexSpectrogram)
        {
            int freqBins = complexSpectrogram.GetLength(0);
            int timeFrames = complexSpectrogram.GetLength(2);

            // Create output array with shape [1, 2*freqBins, timeFrames]
            float[,,] stftFormat = new float[1, 2 * freqBins, timeFrames];

            for (int t = 0; t < timeFrames; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    Complex value = complexSpectrogram[f, 0, t];

                    // Real part goes to first half
                    stftFormat[0, f, t] = (float)value.Real;

                    // Imaginary part goes to second half
                    stftFormat[0, f + freqBins, t] = (float)value.Imaginary;
                }
            }

            return stftFormat;
        }

        /// <summary>
        /// Converts back from float[1, 642, 723] STFT format to Complex[321, 1, 723]
        /// </summary>
        /// <param name="stftFormat">STFT format array with shape [1, 2*freq_bins, time_frames]</param>
        /// <returns>Complex spectrogram with shape [freq_bins, 1, time_frames]</returns>
        public static Complex[,,] ConvertSTFTFormatToComplex(float[,,] stftFormat)
        {
            int totalChannels = stftFormat.GetLength(1);
            int timeFrames = stftFormat.GetLength(2);
            int freqBins = totalChannels / 2;

            Complex[,,] complexSpectrogram = new Complex[freqBins, 1, timeFrames];

            for (int t = 0; t < timeFrames; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    float real = stftFormat[0, f, t];
                    float imag = stftFormat[0, f + freqBins, t];
                    complexSpectrogram[f, 0, t] = new Complex(real, imag);
                }
            }

            return complexSpectrogram;
        }

        public List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]>? statesList = null, int offset = 0)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / 80).ToArray();
            float[] samples = PadHelper.PadSequence(modelInputs);
            Utils.STFTArgs args = new Utils.STFTArgs();
            args.win_len = 640;
            args.fft_len = 640;
            args.win_type = "hanning";
            args.win_inc = 320;
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
                if (name == "inputs")
                {
                    int[] dim = new int[] { batchSize, 642, padSequence.Length / batchSize / 642 };
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
                    float[] output = new float[(int)(samples.Length - sampleRate * 0.5f) / 1];
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
        ~SepProjOfFrcrnSe2()
        {
            Dispose(_disposed);
        }
    }
}
