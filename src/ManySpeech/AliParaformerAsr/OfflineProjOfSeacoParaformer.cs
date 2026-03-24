// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.AliParaformerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr
{
    internal class OfflineProjOfSeacoParaformer : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;
        private InferenceSession? _embedSession;
        private OfflineModel _offlineModel;

        public OfflineProjOfSeacoParaformer(OfflineModel offlineModel)
        {
            _offlineModel = offlineModel;
            _modelSession = offlineModel.ModelSession;
            _embedSession = offlineModel.EmbedSession;
        }
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }

        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<string>? languages = null, List<string>? regions = null)
        {
            ModelOutputEntity modelOutputEntity = ModelProj(modelInputs);
            if (modelOutputEntity != null)
            {
                Tensor<float>? logitsTensor = modelOutputEntity.ModelOut;
                string method = _offlineModel.Method;
                // 2. 根据解码策略执行对应逻辑
                if (string.Equals(method, "greedy", StringComparison.OrdinalIgnoreCase))
                {
                    // 调用对齐原逻辑的贪心搜索
                    ExecuteGreedySearch(logitsTensor, tokenIdsList, timestampsList);
                }
                else if (string.Equals(method, "beam", StringComparison.OrdinalIgnoreCase))
                {
                    // 调用束搜索
                    ExecuteBeamSearch(logitsTensor, tokenIdsList, timestampsList, _offlineModel.BeamWidth);
                }
                else
                {
                    throw new ArgumentException($"Unsupported decode method: {method}, only 'greedy' or 'beam' is allowed");
                }
                if (modelOutputEntity.CifPeak != null)
                {
                    timestampsList = new List<List<int[]>>();
                    Tensor<float> cifPeak = modelOutputEntity.CifPeak;
                    for (int i = 0; i < cifPeak.Dimensions[0]; i++)
                    {
                        float[] usCifPeak = new float[cifPeak.Dimensions[1]];
                        Array.Copy(cifPeak.ToArray(), i * usCifPeak.Length, usCifPeak, 0, usCifPeak.Length);
                        List<int[]> timestamps = ComputeHelper.TimestampLfr6(usCifPeak, tokenIdsList[i].ToArray());
                        timestampsList.Add(timestamps);
                    }
                }
            }
        }

        /// <summary>
        /// 执行贪心搜索解码
        /// </summary>
        private void ExecuteGreedySearch(Tensor<float>? logitsTensor,
                                               List<List<int>> tokenIdsList,
                                               List<List<int[]>> timestampsList)
        {
            if (logitsTensor == null) return;
            for (int batchIndex = 0; batchIndex < logitsTensor.Dimensions[0]; batchIndex++)
            {
                // 存储单个批次的Token ID序列
                int[] batchTokenIds = new int[logitsTensor.Dimensions[1]];
                // 存储单个批次的Token时间戳
                List<int[]> batchTokenTimestamps = new List<int[]>();
                for (int sequenceStep = 0; sequenceStep < logitsTensor.Dimensions[1]; sequenceStep++)
                {
                    // 当前序列位置的最优Token ID
                    int bestTokenId = 0;
                    // 逐一遍历Token，保留概率更大的Token ID
                    for (int tokenIndex = 1; tokenIndex < logitsTensor.Dimensions[2]; tokenIndex++)
                    {
                        bestTokenId = logitsTensor[batchIndex, sequenceStep, bestTokenId] > logitsTensor[batchIndex, sequenceStep, tokenIndex]
                            ? bestTokenId
                            : tokenIndex;
                    }

                    batchTokenIds[sequenceStep] = bestTokenId;
                    batchTokenTimestamps.Add(new int[] { 0, 0 });
                }

                tokenIdsList.Add(batchTokenIds.ToList());
                timestampsList.Add(batchTokenTimestamps);
            }
        }

        /// <summary>
        /// 执行束搜索解码（核心逻辑：维护Top-N候选序列，选整体最优）
        /// </summary>
        /// <param name="beamWidth">束宽度（越大精度越高，性能越低）</param>
        private void ExecuteBeamSearch(Tensor<float> logitsTensor,
                                             List<List<int>> tokenIdsList,
                                             List<List<int[]>> timestampsList,
                                             int beamWidth)
        {
            for (int batchIdx = 0; batchIdx < logitsTensor.Dimensions[0]; batchIdx++)
            {
                // 初始化束：保存(序列, 累计概率)，初始为空序列
                var beam = new List<(List<int> sequence, float totalProb)> { (new List<int>(), 0f) };

                for (int seqIdx = 0; seqIdx < logitsTensor.Dimensions[1]; seqIdx++)
                {
                    var candidates = new List<(List<int> sequence, float totalProb)>();

                    // 遍历当前束中的所有候选序列
                    foreach (var (currentSeq, currentProb) in beam)
                    {
                        // 为当前序列的下一个位置生成所有可能的Token及概率
                        var tokenProbs = new List<(int tokenId, float prob)>();
                        for (int tokenIdx = 0; tokenIdx < logitsTensor.Dimensions[2]; tokenIdx++)
                        {
                            tokenProbs.Add((tokenIdx, logitsTensor[batchIdx, seqIdx, tokenIdx]));
                        }

                        // 按概率排序，取Top-BeamWidth个Token
                        var topTokens = tokenProbs.OrderByDescending(t => t.prob).Take(beamWidth);
                        foreach (var (tokenId, prob) in topTokens)
                        {
                            // 生成新序列并累加概率（这里用加法，实际可改用对数概率避免下溢）
                            var newSeq = new List<int>(currentSeq) { tokenId };
                            float newProb = currentProb + prob;
                            candidates.Add((newSeq, newProb));
                        }
                    }

                    // 从所有候选中选Top-BeamWidth个，更新束
                    beam = candidates.OrderByDescending(c => c.totalProb).Take(beamWidth).ToList();
                }

                // 取束中概率最大的序列作为最终结果
                var bestSequence = beam.OrderByDescending(b => b.totalProb).First().sequence;
                // 补全序列长度（与贪心搜索结果维度对齐）
                while (bestSequence.Count < logitsTensor.Dimensions[1])
                {
                    bestSequence.Add(0); // 补零
                }

                tokenIdsList.Add(bestSequence);
                // 初始化时间戳
                timestampsList.Add(Enumerable.Repeat(new int[] { 0, 0 }, logitsTensor.Dimensions[1]).ToList());
            }
        }

        public Tensor<float>? EmbedProj(List<int[]>? hotwords)
        {
            if (hotwords == null || hotwords.Count == 0)
            {
                return null;
            }
            //float[] y=new float[0];
            Tensor<float>? hwEmbed = null;
            int numHotwords = hotwords.Count;
            int maxLength = 10;
            int[] hotwords_pad = PadList(hotwords, 0, maxLength);
            var inputMeta = _embedSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hotword")
                {
                    int[] dim = new int[] { numHotwords, 10 };
                    var tensor = new DenseTensor<int>(hotwords_pad, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? results = null;
            try
            {
                results = _embedSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    hwEmbed = resultsArray[0].AsTensor<float>();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("SeACo Paraformer Embed infer failed", ex.InnerException);
            }
            return hwEmbed;
        }
        private int[] PadList(List<int[]> hotwords, int paddingValue, int maxLength = 0)
        {
            List<int[]> hotwordsPadList = new List<int[]>(hotwords);
            if (maxLength == 0)
            {
                maxLength = hotwords.Select(x => x.Length).Max();
            }
            for (int i = 0; i < hotwordsPadList.Count; i++)
            {
                hotwordsPadList[i] = hotwordsPadList[i].Length > maxLength ? hotwordsPadList[i].Take(maxLength).ToArray() : hotwordsPadList[i].Concat(Enumerable.Repeat(paddingValue, maxLength - hotwordsPadList[i].Length)).ToArray();
            }
            int[] hotwordsPad = hotwordsPadList.SelectMany(x => x).ToArray();
            return hotwordsPad;
        }

        public ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Tensor<float>? hotwordsEmbed = null;
            List<int[]>? hotwords = modelInputs.SelectMany(x => x.Hotwords).ToList();
            if (hotwords != null && hotwords?.Count > 0)
            {
                hotwordsEmbed = EmbedProj(hotwords);
            }
            else
            {
                hotwords = _offlineModel.Hotwords;
                hotwordsEmbed = EmbedProj(hotwords);
            }
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _modelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 560 / batchSize, 560 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    int[] speech_lengths = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / 560 / batchSize;
                    }
                    var tensor = new DenseTensor<int>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "bias_embed")
                {
                    int[] dim = new int[] { batchSize, 0, 512 };
                    float[] biasEmbed = new float[0];
                    if (hotwordsEmbed != null)
                    {
                        long _hwEmbedLength = hotwordsEmbed.Length;
                        biasEmbed = new float[_hwEmbedLength * batchSize];
                        List<float[]> ebList = new List<float[]>();
                        for (int n = 0; n < hotwordsEmbed.Dimensions[1]; n++)
                        {
                            float[] eb = new float[10 * 512];
                            for (int j = 0; j < hotwordsEmbed.Dimensions[0]; j++)
                            {
                                int k = hotwordsEmbed.Dimensions[2];
                                Array.Copy(hotwordsEmbed.ToArray(), j * hotwordsEmbed.Dimensions[1] * k + n * k, eb, j * k, k);
                            }
                            ebList.Add(eb);
                        }
                        float[] biasEmbedTemp = ebList.SelectMany(x => x).ToArray(); // hwEmbed.ToArray();// 
                        for (int i = 0; i < batchSize; i++)
                        {
                            Array.Copy(biasEmbedTemp, 0, biasEmbed, i * biasEmbedTemp.Length, biasEmbedTemp.Length);
                        }
                        dim = new int[] { batchSize, biasEmbed.Length / 512 / batchSize, 512 };
                    }
                    var tensor = new DenseTensor<float>(biasEmbed, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            ModelOutputEntity modelOutputEntity = new ModelOutputEntity();
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _modelSession.Run(container);

                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    modelOutputEntity.ModelOut = resultsArray[0].AsTensor<float>();
                    modelOutputEntity.ModelOutLens = resultsArray[1].AsEnumerable<int>().ToArray();
                    if (resultsArray.Length >= 4)
                    {
                        Tensor<float> cifPeak = resultsArray[3].AsTensor<float>();
                        modelOutputEntity.CifPeak = cifPeak;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("ModelProj failed", ex);
            }
            return modelOutputEntity;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_modelSession != null)
                    {
                        _modelSession.Dispose();
                    }
                    if (_embedSession != null)
                    {
                        _embedSession.Dispose();
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
        ~OfflineProjOfSeacoParaformer()
        {
            Dispose(_disposed);
        }
    }
}
