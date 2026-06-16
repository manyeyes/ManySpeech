// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.ASR.Model;
using ManySpeech.ASR.Utils;
using ManySpeech.SeqUnit;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.ASR
{
    internal class OfflineSeacoParaformer : IOffline, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession? _modelSession;
        private InferenceSession? _embedSession;
        private OfflineModel _offlineModel;
        private ITokenizer _tokenizer;
        private int _sampleRate = 16000;
        private int _speechLength = 30;
        private bool _isResizeAudioDuration = false;
        private bool _isPaddingSpeech = false;
        private bool _isSampleScalingRequired = true;
        private string[] _hotwords;
        private List<int[]>? _hotwordIds = new List<int[]>();

        public OfflineSeacoParaformer(OfflineModel offlineModel)
        {
            _offlineModel = offlineModel;
            _modelSession = offlineModel.ModelSession;
            _embedSession = offlineModel.EmbedSession;
            _tokenizer = AutoTokenizer.Create(type: TokenizerType.Textoken, vocabFilePath: offlineModel.TokensFilePath);
            _hotwords = new string[] { "魔搭" };
            if (_offlineModel.Hotwords?.Length > 0)
            {
                _hotwords = _offlineModel.Hotwords;
            }
            if (_hotwords.Length > 0)
            {
                foreach (string word in _hotwords)
                {
                    int[]? ids = _tokenizer.Encode(word);
                    if (ids != null)
                    {
                        _hotwordIds.Add(ids);
                    }
                }
                _hotwordIds.Add(new int[] { OfflineModel.Sos_eos_id });
            }
        }
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }
        public ITokenizer Tokenizer { get => _tokenizer; set => _tokenizer = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public bool IsPaddingSpeech { get => _isPaddingSpeech; set => _isPaddingSpeech = value; }
        public bool IsResizeAudioDuration { get => _isResizeAudioDuration; set => _isResizeAudioDuration = value; }
        public bool IsSampleScalingRequired { get => _isSampleScalingRequired; set => _isSampleScalingRequired = value; }

        public List<int> GetDecoderInitTokenIds()
        {
            return new List<int> { OfflineModel.Blank_id, OfflineModel.Blank_id };
        }

        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<string>? languages = null, List<string>? regions = null)
        {
            ModelOutputEntity modelOutputEntity = ModelProj(modelInputs);
            if (modelOutputEntity != null)
            {
                Tensor<float>? logitsTensor = modelOutputEntity.ModelOut;
                string method = _offlineModel.Method;
                // Execute the corresponding logic according to the decoding strategy
                if (string.Equals(method, "greedy", StringComparison.OrdinalIgnoreCase))
                {
                    // Invoke greedy search aligned with the original logic
                    ExecuteGreedySearch(logitsTensor, tokenIdsList, timestampsList);
                }
                else if (string.Equals(method, "beam", StringComparison.OrdinalIgnoreCase))
                {
                    // Invoke beam search
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
        /// Executes greedy search decoding
        /// </summary>
        private void ExecuteGreedySearch(Tensor<float>? logitsTensor,
                                               List<List<int>> tokenIdsList,
                                               List<List<int[]>> timestampsList)
        {
            if (logitsTensor == null) return;
            for (int batchIndex = 0; batchIndex < logitsTensor.Dimensions[0]; batchIndex++)
            {
                // Store the Token ID sequence for a single batch
                int[] batchTokenIds = new int[logitsTensor.Dimensions[1]];
                // Store the Token timestamps for a single batch
                List<int[]> batchTokenTimestamps = new List<int[]>();
                for (int sequenceStep = 0; sequenceStep < logitsTensor.Dimensions[1]; sequenceStep++)
                {
                    // The optimal Token ID at the current sequence position
                    int bestTokenId = 0;
                    // Iterate through Tokens one by one and retain Token IDs with higher probabilities
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
        /// Executes beam search decoding (maintains Top-N candidate sequences and selects the globally optimal one)
        /// </summary>
        /// <param name="beamWidth">Beam width (higher = better accuracy, lower performance)</param>
        private void ExecuteBeamSearch(Tensor<float> logitsTensor,
                                             List<List<int>> tokenIdsList,
                                             List<List<int[]>> timestampsList,
                                             int beamWidth)
        {
            for (int batchIdx = 0; batchIdx < logitsTensor.Dimensions[0]; batchIdx++)
            {
                // Initialize beam: stores (sequence, cumulative probability), starting with empty sequence
                var beam = new List<(List<int> sequence, float totalProb)> { (new List<int>(), 0f) };

                for (int seqIdx = 0; seqIdx < logitsTensor.Dimensions[1]; seqIdx++)
                {
                    var candidates = new List<(List<int> sequence, float totalProb)>();

                    // Iterate all candidate sequences in the current beam
                    foreach (var (currentSeq, currentProb) in beam)
                    {
                        // Generate all possible Tokens and their probabilities for the next position of the current sequence
                        var tokenProbs = new List<(int tokenId, float prob)>();
                        for (int tokenIdx = 0; tokenIdx < logitsTensor.Dimensions[2]; tokenIdx++)
                        {
                            tokenProbs.Add((tokenIdx, logitsTensor[batchIdx, seqIdx, tokenIdx]));
                        }

                        // Sort by probability and take top BeamWidth Tokens
                        var topTokens = tokenProbs.OrderByDescending(t => t.prob).Take(beamWidth);
                        foreach (var (tokenId, prob) in topTokens)
                        {
                            // Create new sequence and accumulate probability (addition used here; log probability can be used to avoid underflow in practice)
                            var newSeq = new List<int>(currentSeq) { tokenId };
                            float newProb = currentProb + prob;
                            candidates.Add((newSeq, newProb));
                        }
                    }

                    // Select top BeamWidth candidates and update the beam
                    beam = candidates.OrderByDescending(c => c.totalProb).Take(beamWidth).ToList();
                }

                // Select the sequence with the highest probability as the final result
                var bestSequence = beam.OrderByDescending(b => b.totalProb).First().sequence;
                // Pad sequence length to match dimensions with greedy search results
                while (bestSequence.Count < logitsTensor.Dimensions[1])
                {
                    bestSequence.Add(0); // Pad with zero
                }

                tokenIdsList.Add(bestSequence);
                // Initialize timestamps
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

            var hotwordList = modelInputs.SelectMany(x => x.Hotwords).ToList();
            string[]? hotwords = hotwordList.Count > 0 ? hotwordList.ToArray() : null;
            List<int[]>? hotwordIds = new List<int[]>();
            if (hotwords != null && hotwords.Length > 0)
            {
                foreach (string word in hotwords)
                {
                    int[]? ids = _tokenizer.Encode(word);
                    if (ids != null)
                    {
                        hotwordIds.Add(ids);
                    }
                }
                hotwordIds.Add(new int[] { OfflineModel.Sos_eos_id });
            }
            else
            {
                hotwordIds = _hotwordIds;
            }
            hotwordsEmbed = EmbedProj(hotwordIds);

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
        ~OfflineSeacoParaformer()
        {
            Dispose(_disposed);
        }
    }
}
