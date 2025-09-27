using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using ManySpeech.MoonshineAsr.Model;
using ManySpeech.MoonshineAsr.Utils;

namespace ManySpeech.MoonshineAsr
{
    internal class OfflineProj : IOfflineProj
    {
        private InferenceSession _preprocessSession;
        private InferenceSession _encodeSession;
        private InferenceSession _cachedDecodeSession;
        private InferenceSession _uncachedDecodeSession;
        private CustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _sos_eos_id = 0;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;
        public OfflineProj(AsrModel asrModel)
        {
            _preprocessSession = asrModel.PreprocessSession;
            _encodeSession = asrModel.EncodeSession;
            _cachedDecodeSession = asrModel.CachedDecodeSession;
            _uncachedDecodeSession = asrModel.UncachedDecodeSession;
            _blank_id = asrModel.Blank_id;
            _sos_eos_id = asrModel.Sos_eos_id;
            _unk_id = asrModel.Unk_id;
            _featureDim = asrModel.FeatureDim;
            _sampleRate = asrModel.SampleRate;
            _customMetadata = asrModel.CustomMetadata;
            _chunkLength = asrModel.ChunkLength;
            _shiftLength = asrModel.ShiftLength;
            _required_cache_size = asrModel.Required_cache_size;
        }

        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public InferenceSession PreprocessSession { get => _preprocessSession; set => _preprocessSession = value; }
        public InferenceSession EncodeSession { get => _encodeSession; set => _encodeSession = value; }
        public InferenceSession CachedDecodeSession { get => _cachedDecodeSession; set => _cachedDecodeSession = value; }
        public InferenceSession UncachedDecodeSession { get => _uncachedDecodeSession; set => _uncachedDecodeSession = value; }

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

        public PreprocessOutputEntity PreprocessProj(List<AsrInputEntity> modelInputs)
        {
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _preprocessSession.InputMetadata;
            PreprocessOutputEntity preprocessOutputEntity = new PreprocessOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "args_0")
                {
                    int[] dim = new int[] { 1, padSequence.Length };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> preprocessResults = null;
                preprocessResults = _preprocessSession.Run(container);

                if (preprocessResults != null)
                {
                    var preprocessResultsArray = preprocessResults.ToArray();
                    var outputTensor = preprocessResultsArray[0].AsTensor<float>();
                    preprocessOutputEntity.SeqDim = outputTensor.Dimensions.ToArray();
                    preprocessOutputEntity.Seq = outputTensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return preprocessOutputEntity;
        }

        public EncodeOutputEntity EncodeProj(PreprocessOutputEntity preprocessOutputEntity)
        {
            float[]? padSequence = preprocessOutputEntity.Seq;
            int batchSize = preprocessOutputEntity.SeqDim[0];
            int seqLength = preprocessOutputEntity.SeqDim[1];
            var inputMeta = _encodeSession.InputMetadata;
            EncodeOutputEntity encodeOutputEntity = new EncodeOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "args_0")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / batchSize / 288, 288 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "args_1")
                {
                    int[] dim = new int[] { batchSize };
                    int[] seqLengths = new int[batchSize];
                    for (int i = 0; i < seqLengths.Length; i++)
                    {
                        seqLengths[i] = seqLength;
                    }
                    var tensor = new DenseTensor<int>(seqLengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encodeSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    var outputTensor = encoderResultsArray[0].AsTensor<float>();
                    encodeOutputEntity.Output = outputTensor.ToArray();
                    encodeOutputEntity.OutputDim = outputTensor.Dimensions.ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return encodeOutputEntity;
        }

        public UncachedDecodeOutputEntity UncachedDecodeProj(EncodeOutputEntity encodeOutputEntity, List<int[]> nextTokens, List<int> seqLens)
        {
            int batchSize = nextTokens.Count;
            List<int[]> inputs = nextTokens;
            int[] seqLengths = seqLens.ToArray();
            CustomMetadata customMetadata = _customMetadata;
            UncachedDecodeOutputEntity uncachedDecodeOutputEntity = new UncachedDecodeOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _uncachedDecodeSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "args_0")
                {
                    int[] dim = new int[] { inputs.Count, 1 };
                    var tensor = new DenseTensor<int>(inputs.SelectMany(x => x).ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "args_1")
                {
                    float[] context = encodeOutputEntity.Output;
                    int[] dim = encodeOutputEntity.OutputDim;//new int[] { 1, context.Length / 288, 288 };
                    var tensor = new DenseTensor<float>(context, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "args_2")
                {
                    int[] dim = new int[] { seqLens.Count };
                    var tensor = new DenseTensor<int>(seqLengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _uncachedDecodeSession.Run(container);

                List<float> rescoring_score = new List<float>();

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    var reversibleEmbedding = decoderResultsArray[0].AsTensor<float>();
                    uncachedDecodeOutputEntity.ReversibleEmbedding = reversibleEmbedding;
                    uncachedDecodeOutputEntity.CacheList = new List<float[]>();
                    for (int i = 1; i < decoderResultsArray.Length; i++)
                    {
                        uncachedDecodeOutputEntity.CacheList.Add(decoderResultsArray[i].AsTensor<float>().ToArray());
                        decoderResultsArray[i].Dispose();
                    }

                }
            }
            catch (Exception ex)
            {
                //
            }
            return uncachedDecodeOutputEntity;
        }

        public CachedDecodeOutputEntity CachedDecodeProj(EncodeOutputEntity encodeOutputEntity, List<int[]> nextTokens, List<int> seqLens, List<float[]> cacheList)
        {
            int batchSize = nextTokens.Count;
            List<int[]> inputs = nextTokens;
            int[] seqLengths = seqLens.ToArray();
            CustomMetadata customMetadata = _customMetadata;
            CachedDecodeOutputEntity cachedDecodeOutputEntity = new CachedDecodeOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _cachedDecodeSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "args_0")
                {
                    int[] dim = new int[] { inputs.Count, 1 };
                    var tensor = new DenseTensor<int>(inputs.SelectMany(x => x).ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "args_1")
                {
                    float[] context = encodeOutputEntity.Output;
                    int[] dim = encodeOutputEntity.OutputDim;//new int[] { 1, context.Length / 288, 288 };
                    var tensor = new DenseTensor<float>(context, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "args_2")
                {
                    int[] dim = new int[] { seqLens.Count };
                    var tensor = new DenseTensor<int>(seqLengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }
            for (int i = 0; i < cacheList.Count; i++)
            {
                string name = "args_" + (i + 3).ToString();
                if (inputMeta.Keys.Contains(name))
                {
                    float[] cache = cacheList[i];
                    int[] dim = new int[] { 1, cache.Length / 8 / 36, 8, 36 };
                    var tensor = new DenseTensor<float>(cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _cachedDecodeSession.Run(container);

                List<float> rescoring_score = new List<float>();

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    var reversibleEmbedding = decoderResultsArray[0].AsTensor<float>();
                    cachedDecodeOutputEntity.ReversibleEmbedding = reversibleEmbedding;
                    cachedDecodeOutputEntity.CacheList = new List<float[]>();
                    for (int i = 1; i < decoderResultsArray.Length; i++)
                    {
                        cachedDecodeOutputEntity.CacheList.Add(decoderResultsArray[i].AsTensor<float>().ToArray());
                        decoderResultsArray[i].Dispose();
                    }
                }
            }
            catch (Exception ex)
            {
                //
            }
            return cachedDecodeOutputEntity;
        }
    }
}
