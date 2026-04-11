using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using ManySpeech.FireRedAsr.Model;
using ManySpeech.FireRedAsr.Utils;
using System;

namespace ManySpeech.FireRedAsr
{
    internal class OfflineProjOfFireRedASRAEDKV : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _ctcSession;
        private CustomMetadata _customMetadata;
        private ConfEntity _confEntity;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _pad_id = 2;
        private int _sos_id = 3;
        private int _eos_id = 4;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;

        private int _maxNewTokens = 256;
        private const int D_MODEL = 1280;
        private const int N_LAYERS = 16;
        public OfflineProjOfFireRedASRAEDKV(OfflineModel offlineModel)
        {
            _encoderSession = offlineModel.EncoderSession;
            _decoderSession = offlineModel.DecoderSession;
            _ctcSession = offlineModel.CtcSession;
            _blank_id = offlineModel.Blank_id;
            _unk_id = offlineModel.Unk_id;
            _pad_id = offlineModel.Pad_id;
            _sos_id = offlineModel.Sos_id;
            _eos_id = offlineModel.Eos_id;
            _featureDim = offlineModel.FeatureDim;
            _sampleRate = offlineModel.SampleRate;
            _customMetadata = offlineModel.CustomMetadata;
            _chunkLength = offlineModel.ChunkLength;
            _shiftLength = offlineModel.ShiftLength;
            _required_cache_size = offlineModel.Required_cache_size;
            _confEntity = offlineModel.ConfEntity;
        }
        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession CtcSession { get => _ctcSession; set => _ctcSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int Pad_id { get => _pad_id; set => _pad_id = value; }
        public int Sos_id { get => _sos_id; set => _sos_id = value; }
        public int Eos_id { get => _eos_id; set => _eos_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public ConfEntity ConfEntity { get => _confEntity; set => _confEntity = value; }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            int batchSize = statesList.Count;
            Debug.Assert(statesList[0].Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = statesList[0].Count;
            for (int i = 0; i < fsmnLayer; i++)
            {
                float[] statesItemTemp = new float[statesList[0][i].Length * batchSize];
                int statesItemTemp_item_length = statesList[0][i].Length;
                int statesItemTemp_item_axisnum = 1280;
                for (int x = 0; x < statesItemTemp_item_length / statesItemTemp_item_axisnum; x++)
                {
                    for (int n = 0; n < batchSize; n++)
                    {
                        float[] statesItemTemp_item = statesList[n][0];
                        Array.Copy(statesItemTemp_item, x * statesItemTemp_item_axisnum, statesItemTemp, (x * batchSize + n) * statesItemTemp_item_axisnum, statesItemTemp_item_axisnum);
                    }
                }
                states.Add(statesItemTemp);
            }
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = states.Count;
            int batchSize = states[0].Length / 1280;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 1280;
                    int statesItemTemp_size = 1 * 1280;
                    float[] statesItemTemp_item = new float[statesItemTemp_size];
                    for (int k = 0; k < statesItemTemp_size / statesItemTemp_axisnum; k++)
                    {
                        Array.Copy(item, (item.Length / statesItemTemp_size * k + b) * statesItemTemp_axisnum, statesItemTemp_item, k * statesItemTemp_axisnum, statesItemTemp_axisnum);
                    }
                    statesListItem.Add(statesItemTemp_item);
                }
                statesList.Add(statesListItem);
            }
            return statesList;
        }

        public EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            Int64[] inputLengths = new Int64[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                inputLengths[i] = (long)padSequence.Length / batchSize / 80;
            }
            var inputMeta = _encoderSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 80 / batchSize, 80 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "input_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<Int64>(inputLengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
            }

            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            try
            {
                using var encoderResults = _encoderSession.Run(container);
                var resultsArray = encoderResults.ToArray();
                // 前三个输出：output, output_lengths, mask
                encoderOutput.EncOut = resultsArray[0].AsTensor<float>();
                encoderOutput.EncOutLens = resultsArray[1].AsTensor<Int64>().ToArray();
                encoderOutput.Mask = resultsArray[2].AsTensor<bool>().ToArray();

                // 后续输出：cross_k_0, cross_v_0, cross_k_1, cross_v_1, ...
                List<float[]> crossKvList = new List<float[]>();
                for (int i = 3; i < resultsArray.Length; i++)
                {
                    var tensor = resultsArray[i].AsTensor<float>();
                    crossKvList.Add(tensor.ToArray());
                }
                encoderOutput.CrossKVList = crossKvList;
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }

        public (List<List<int>> tokens, List<float[]> newCaches, List<int> cacheLengths) DecoderProj(
            List<List<int>> initialTokens,
            bool[] src_mask, List<float[]> crossKVList,
            List<float[]> stackedSelfCaches,   // self KV 缓存，形状 [batch*cur_len*D_MODEL] 每层
            int batchSize
        )
        {
            // 确保初始 tokens 包含 SOS
            for (int b = 0; b < batchSize; b++)
                if (initialTokens[b] == null || initialTokens[b].Count == 0)
                    initialTokens[b] = new List<int> { _sos_id };

            List<List<int>> tokensList = initialTokens.Select(t => new List<int>(t)).ToList();
            List<int> steps = tokensList.Select(t => t.Count - 1).ToList();
            List<int> cacheLengths = new List<int>(batchSize);
            int totalLen = stackedSelfCaches.Count > 0 ? stackedSelfCaches[0].Length : 0;
            int curCacheLen = batchSize > 0 ? totalLen / (batchSize * D_MODEL) : 0;
            for (int b = 0; b < batchSize; b++)
                cacheLengths.Add(curCacheLen);

            int encSeqLen = crossKVList[0].Length / (batchSize * D_MODEL);  // 每个 cross_k 数组长度
            int srcLen = src_mask.Length / batchSize;

            int maxNewTokens = _maxNewTokens;
            bool[] finished = new bool[batchSize];

            for (int step = 0; step < maxNewTokens; step++)
            {
                if (finished.All(f => f)) break;

                var container = new List<NamedOnnxValue>();

                // token: (batch, 1)
                long[] curTokens = new long[batchSize];
                for (int b = 0; b < batchSize; b++)
                    curTokens[b] = finished[b] ? _eos_id : tokensList[b].Last();
                container.Add(NamedOnnxValue.CreateFromTensor("token", new DenseTensor<long>(curTokens, new[] { batchSize, 1 })));

                // step
                long stepVal = steps[0];
                container.Add(NamedOnnxValue.CreateFromTensor("step", new DenseTensor<long>(new long[] { stepVal }, new[] { 1 })));

                // src_mask: (batch, 1, srcLen)
                container.Add(NamedOnnxValue.CreateFromTensor("src_mask", new DenseTensor<bool>(src_mask, new[] { batchSize, 1, srcLen })));

                // 输入 self KV 缓存和 cross KV 缓存
                for (int i = 0; i < N_LAYERS; i++)
                {
                    // self_k, self_v
                    float[] selfK = stackedSelfCaches[2 * i];
                    float[] selfV = stackedSelfCaches[2 * i + 1];
                    int cacheLen = cacheLengths[0];
                    container.Add(NamedOnnxValue.CreateFromTensor($"self_k_cache_{i}", new DenseTensor<float>(selfK, new[] { batchSize, cacheLen, D_MODEL })));
                    container.Add(NamedOnnxValue.CreateFromTensor($"self_v_cache_{i}", new DenseTensor<float>(selfV, new[] { batchSize, cacheLen, D_MODEL })));

                    // cross_k, cross_v
                    float[] crossK = crossKVList[2 * i];
                    float[] crossV = crossKVList[2 * i + 1];
                    container.Add(NamedOnnxValue.CreateFromTensor($"cross_k_{i}", new DenseTensor<float>(crossK, new[] { batchSize, encSeqLen, D_MODEL })));
                    container.Add(NamedOnnxValue.CreateFromTensor($"cross_v_{i}", new DenseTensor<float>(crossV, new[] { batchSize, encSeqLen, D_MODEL })));
                }

                using var results = _decoderSession.Run(container);
                var resultDict = results.ToDictionary(r => r.Name);

                // 获取 logits
                var logitsTensor = resultDict["logits"].AsTensor<float>();
                int vocabSize = logitsTensor.Dimensions[1];

                // 更新 self KV 缓存
                List<float[]> newStackedSelfCaches = new List<float[]>();
                for (int i = 0; i < N_LAYERS; i++)
                {
                    var newSelfK = resultDict[$"new_self_k_cache_{i}"].AsTensor<float>().ToArray();
                    var newSelfV = resultDict[$"new_self_v_cache_{i}"].AsTensor<float>().ToArray();
                    newStackedSelfCaches.Add(newSelfK);
                    newStackedSelfCaches.Add(newSelfV);
                }
                stackedSelfCaches = newStackedSelfCaches;
                for (int b = 0; b < batchSize; b++)
                    cacheLengths[b]++;

                // 贪心解码
                int[] nextTokens = new int[batchSize];
                for (int b = 0; b < batchSize; b++)
                {
                    if (finished[b])
                    {
                        nextTokens[b] = _eos_id;
                        continue;
                    }
                    int bestIdx = 0;
                    float maxLogit = logitsTensor[b, 0];
                    for (int j = 1; j < vocabSize; j++)
                        if (logitsTensor[b, j] > maxLogit) { maxLogit = logitsTensor[b, j]; bestIdx = j; }
                    nextTokens[b] = bestIdx;
                }

                for (int b = 0; b < batchSize; b++)
                {
                    if (!finished[b])
                    {
                        tokensList[b].Add(nextTokens[b]);
                        if (nextTokens[b] == _eos_id)
                            finished[b] = true;
                    }
                    steps[b]++;
                }
            }

            return (tokensList, stackedSelfCaches, cacheLengths);
        }
        public DecoderOutputEntity DecoderProj(List<List<int>> tokensList, float[] encoder_outputs, bool[] src_mask, List<float[]> cacheList)
        {
            return null;
        }

        public CtcOutputEntity CtcProj(float[] encoder_outputs, int batchSize = 1)
        {
            CustomMetadata customMetadata = _customMetadata;
            CtcOutputEntity ctcOutputEntity = new CtcOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _ctcSession.InputMetadata;
            try
            {
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "encoder_outputs")
                    {
                        int[] dim = new int[3] { batchSize, encoder_outputs.Length / 1280 / batchSize, 1280 };
                        var tensor = new DenseTensor<float>(encoder_outputs, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> ctcResults = null;
                ctcResults = _ctcSession.Run(container);

                List<float> rescoring_score = new List<float>();
                if (ctcResults != null)
                {
                    var ctcResultsArray = ctcResults.ToArray();
                    Tensor<float> logits_tensor = ctcResultsArray[0].AsTensor<float>();
                    List<List<float[]>> logitsList = new List<List<float[]>>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        List<float[]> item = new List<float[]>();
                        int t = logits_tensor.Dimensions[1];
                        for (int j = 0; j < t; j++)
                        {
                            int n = logits_tensor.Dimensions[2];
                            float[] row = new float[n];
                            for (int k = 0; k < n; k++)
                            {
                                row[k] = logits_tensor[i, j, k];
                            }
                            item.Add(row);
                        }
                        logitsList.Add(item);
                    }
                    ctcOutputEntity.LogitsList = logitsList;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return ctcOutputEntity;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_encoderSession != null)
                    {
                        _encoderSession.Dispose();
                    }
                    if (_decoderSession != null)
                    {
                        _decoderSession.Dispose();
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
        ~OfflineProjOfFireRedASRAEDKV()
        {
            Dispose(_disposed);
        }
    }
}
