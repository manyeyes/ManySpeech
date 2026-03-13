// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.AudioTagging.Model;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AudioTagging
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2025 by manyeyes
    /// </summary>
    public class OfflineTagging : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private int _topK = 3;
        private IOfflineProj _offlineProj;

        public OfflineTagging(string modelFilePath, string tokensFilePath, int topK = 3, string configFilePath = "", int threadsNum = 1)
        {
            OfflineModel offlineModel = new OfflineModel(modelFilePath: modelFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _topK = topK;
            _offlineProj = new OfflineProjOfCED(offlineModel);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_offlineProj);
            return onlineStream;
        }
        public OfflineTaggingResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            OfflineTaggingResultEntity offlineRecognizerResultEntity = GetResults(streams)[0];

            return offlineRecognizerResultEntity;
        }
        public List<OfflineTaggingResultEntity> GetResults(List<OfflineStream> streams)
        {
            this.Forward(streams);
            List<OfflineTaggingResultEntity> offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            List<ModelInputEntity> modelInputs = new List<ModelInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<List<int>> tokensList = new List<List<int>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                ModelInputEntity modelInputEntity = new ModelInputEntity();

                modelInputEntity.Speech = stream.GetDecodeChunk();
                if (modelInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                modelInputEntity.SpeechLength = modelInputEntity.Speech.Length;
                modelInputs.Add(modelInputEntity);
                statesList.Add(stream.States);
                tokensList.Add(stream.Tokens);
                streamsWorking.Add(stream);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OfflineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                ModelOutputEntity modelOutputEntity = _offlineProj.ModelProj(modelInputs);
                if (modelOutputEntity != null)
                {
                    Tensor<float>? logitsTensor = modelOutputEntity.ModelOut;
                    int topK = _topK;
                    var (batchIndices, batchProbs) = ParseTensorTopKResults(logitsTensor, topK);
                    int streamIndex = 0;
                    foreach (OfflineStream stream in streams)
                    {
                        stream.Tokens = batchIndices[streamIndex].ToList();
                        stream.Probs = batchProbs[streamIndex].ToList();
                        stream.RemoveDecodedChunk();
                        streamIndex++;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }
        }

        /// <summary>
        /// 解析 Tensor 类型的批量 logits，获取每个样本的 top-k 结果
        /// </summary>
        /// <param name="logitsTensor">ONNX Runtime 输出的 Tensor，形状为 [batchSize, classCount]</param>
        /// <param name="topK">每个样本需要提取的 top-k 数量</param>
        /// <returns>元组：二维数组（[batchSize, topK]）- 每个样本的 top-k 索引；二维数组（[batchSize, topK]）- 每个样本的 top-k 概率</returns>
        /// <exception cref="ArgumentNullException">logitsTensor 为空时抛出</exception>
        /// <exception cref="ArgumentException">Tensor 维度错误（非 2 维）或类别数非 classCount 时抛出</exception>
        /// <exception cref="ArgumentOutOfRangeException">topK 无效时抛出</exception>
        public (int[][] topKIndicesBatch, double[][] topKProbabilitiesBatch) ParseTensorTopKResults(
            Tensor<float> logitsTensor, int topK)
        {
            // 1. 参数校验
            if (logitsTensor == null)
            {
                throw new ArgumentNullException(nameof(logitsTensor), "Logits Tensor 不能为空");
            }

            // 校验 Tensor 维度：必须是 2 维
            if (logitsTensor.Rank != 2)
            {
                throw new ArgumentException($"Logits Tensor 必须是 2 维，当前维度：{logitsTensor.Rank}", nameof(logitsTensor));
            }

            // 校验类别数：第二维必须与类别数匹配
            int classCount = logitsTensor.Dimensions[1];
            if (classCount != _tokens.Length)
            {
                throw new ArgumentException($"Logits Tensor 第二维与类别数不匹配，当前：{classCount}", nameof(logitsTensor));
            }

            if (topK <= 0 || topK > classCount)
            {
                throw new ArgumentOutOfRangeException(nameof(topK),
                    $"topK 必须大于 0 且不超过类别数，当前：{topK}");
            }

            // 2. 获取 batch 大小，并将 Tensor 转换为 C# 二维数组
            int batchSize = logitsTensor.Dimensions[0];
            double[][] logitsArray = ConvertTensorTo2DArray(logitsTensor, batchSize, classCount);

            // 3. 复用批量解析逻辑（和之前的二维数组解析逻辑一致）
            return ParseBatchTopKResults(logitsArray, topK);
        }

        /// <summary>
        /// 将 ONNX Runtime 的 Float Tensor 转换为 C# 二维 double 数组
        /// </summary>
        /// <param name="tensor">输入 Tensor（float 类型）</param>
        /// <param name="batchSize">批量大小</param>
        /// <param name="classCount">类别数（classCount）</param>
        /// <returns>二维 double 数组 [batchSize, classCount]</returns>
        private static double[][] ConvertTensorTo2DArray(Tensor<float> tensor, int batchSize, int classCount)
        {
            double[][] result = new double[batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                result[i] = new double[classCount];
                for (int j = 0; j < classCount; j++)
                {
                    // Tensor 取值：先 batch 索引，再类别索引；转换为 double 提升精度
                    result[i][j] = tensor[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// 解析二维数组形式的批量 logits（复用之前的逻辑）
        /// </summary>
        /// <param name="logits">批量 logits 数组 [batchSize, classCount]</param>
        /// <param name="topK">top-k 数量</param>
        /// <returns>批量 top-k 索引和概率</returns>
        private static (int[][] topKIndicesBatch, double[][] topKProbabilitiesBatch) ParseBatchTopKResults(double[][] logits, int topK)
        {
            int batchSize = logits.Length;
            int[][] topKIndicesBatch = new int[batchSize][];
            double[][] topKProbabilitiesBatch = new double[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                var (indices, probs) = ParseSingleSampleTopK(logits[i], topK);
                topKIndicesBatch[i] = indices;
                topKProbabilitiesBatch[i] = probs;
            }

            return (topKIndicesBatch, topKProbabilitiesBatch);
        }

        /// <summary>
        /// 解析单个样本的 logits 数组
        /// </summary>
        private static (int[] topKIndices, double[] topKProbabilities) ParseSingleSampleTopK(double[] singleSampleLogits, int topK)
        {
            var sortedPairs = Enumerable.Range(0, singleSampleLogits.Length)
                .Select(idx => new { Index = idx, Value = singleSampleLogits[idx] })
                .OrderByDescending(pair => pair.Value)
                .Take(topK)
                .ToArray();
            return (
                sortedPairs.Select(pair => pair.Index).ToArray(),
                sortedPairs.Select(pair => pair.Value).ToArray()
            );
            //var sortedPairs = singleSampleLogits
            //    .Select((value, idx) => new { value, idx })
            //    .OrderByDescending(x => x.value)
            //    .Take(topK)
            //    .ToList();

            //return (
            //    sortedPairs.Select(pair => pair.idx).ToArray(),
            //    sortedPairs.Select(pair => pair.value).ToArray()
            //);
        }

        private List<OfflineTaggingResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineTaggingResultEntity> offlineTaggingResultEntities = new List<OfflineTaggingResultEntity>();
            List<string> text_results = new List<string>();
#pragma warning disable CS8602 // 解引用可能出现空引用。

            foreach (var stream in streams)
            {
                OfflineTaggingResultEntity offlineTaggingResultEntity = new OfflineTaggingResultEntity();
                string lastToken = "";
#if NET6_0_OR_GREATER
                foreach (var result in stream.Tokens)
                {
                    Int64 token = result;
#else
                for (int i = 0; i < stream.Tokens.Count; i++)
                {
                    Int64 token = stream.Tokens[i];
#endif
                    if (token == 2)
                    {
                        break;
                    }
                    string currText = _tokens[token];
                    if (currText != "</s>" && currText != "<s>" && currText != "<sos/eos>" && currText != "<blank>" && currText != "<unk>" && currText != "<sos>" && currText != "<eos>" && currText != "<pad>")
                    {
                        offlineTaggingResultEntity.Taggings.Add(currText);
                    }
                    else
                    {
                        offlineTaggingResultEntity.Taggings.Add("");
                    }
                }
                offlineTaggingResultEntity.Tokens = stream.Tokens;
                offlineTaggingResultEntity.Probs = stream.Probs;
                offlineTaggingResultEntity.Tagging = offlineTaggingResultEntity.Taggings?.First() ?? "";
                offlineTaggingResultEntities.Add(offlineTaggingResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。
            return offlineTaggingResultEntities;
        }

        public void DisposeOfflineStream(OfflineStream offlineStream)
        {
            if (offlineStream != null)
            {
                offlineStream.Dispose();
            }
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_offlineProj != null)
                    {
                        _offlineProj.Dispose();
                    }
                    if (_tokens != null)
                    {
                        _tokens = null;
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
        ~OfflineTagging()
        {
            Dispose(_disposed);
        }
    }
}