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
        /// Parses batch logits from a Tensor to obtain top-k results for each sample.
        /// </summary>
        /// <param name="logitsTensor">ONNX Runtime output Tensor with shape [batchSize, classCount].</param>
        /// <param name="topK">Number of top-k results to extract per sample.</param>
        /// <returns>Tuple: 2D array ([batchSize, topK]) - top-k indices per sample; 2D array ([batchSize, topK]) - top-k probabilities per sample.</returns>
        /// <exception cref="ArgumentNullException">Thrown when logitsTensor is null.</exception>
        /// <exception cref="ArgumentException">Thrown when tensor rank is not 2 or the number of classes does not match.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is invalid.</exception>
        public (int[][] topKIndicesBatch, double[][] topKProbabilitiesBatch) ParseTensorTopKResults(
            Tensor<float> logitsTensor, int topK)
        {
            // 1. Parameter validation
            if (logitsTensor == null)
            {
                throw new ArgumentNullException(nameof(logitsTensor), "Logits tensor cannot be null.");
            }

            // Validate tensor rank: must be 2
            if (logitsTensor.Rank != 2)
            {
                throw new ArgumentException($"Logits tensor must be 2-dimensional. Current rank: {logitsTensor.Rank}", nameof(logitsTensor));
            }

            // Validate number of classes: second dimension must match the number of classes
            int classCount = logitsTensor.Dimensions[1];
            if (classCount != _tokens.Length)
            {
                throw new ArgumentException($"The second dimension of the logits tensor does not match the number of classes. Current: {classCount}", nameof(logitsTensor));
            }

            if (topK <= 0 || topK > classCount)
            {
                throw new ArgumentOutOfRangeException(nameof(topK),
                    $"topK must be greater than 0 and not exceed the number of classes. Current: {topK}");
            }

            // 2. Get batch size and convert the tensor to a C# 2D array
            int batchSize = logitsTensor.Dimensions[0];
            double[][] logitsArray = ConvertTensorTo2DArray(logitsTensor, batchSize, classCount);

            // 3. Reuse the batch parsing logic (consistent with the previous 2D array parsing logic)
            return ParseBatchTopKResults(logitsArray, topK);
        }

        /// <summary>
        /// Converts an ONNX Runtime Float Tensor to a C# 2D double array.
        /// </summary>
        /// <param name="tensor">Input tensor (float type).</param>
        /// <param name="batchSize">Batch size.</param>
        /// <param name="classCount">Number of classes.</param>
        /// <returns>2D double array [batchSize, classCount].</returns>
        private static double[][] ConvertTensorTo2DArray(Tensor<float> tensor, int batchSize, int classCount)
        {
            double[][] result = new double[batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                result[i] = new double[classCount];
                for (int j = 0; j < classCount; j++)
                {
                    // Tensor access: batch index first, then class index; convert to double for precision
                    result[i][j] = tensor[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Parses batch logits from a 2D array (reusing previous logic).
        /// </summary>
        /// <param name="logits">Batch logits array [batchSize, classCount].</param>
        /// <param name="topK">Number of top-k results.</param>
        /// <returns>Batch top-k indices and probabilities.</returns>
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
        /// Parses a single sample's logits array.
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