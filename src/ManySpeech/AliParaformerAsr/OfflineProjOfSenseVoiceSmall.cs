// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.AliParaformerAsr.Utils;
using ManySpeech.SeqUnit;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr
{
    internal class OfflineProjOfSenseVoiceSmall : IOfflineProj, IDisposable
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

        private bool _useITN = false;
        private string _textnorm = "woitn";
        private Dictionary<string, int> _lidDict = new Dictionary<string, int>() { { "auto", 0 }, { "zh", 3 }, { "en", 4 }, { "yue", 7 }, { "ja", 11 }, { "ko", 12 }, { "nospeech", 13 } };
        private Dictionary<int, int> _lidIntDict = new Dictionary<int, int>() { { 24884, 3 }, { 24885, 4 }, { 24888, 7 }, { 24892, 11 }, { 24896, 12 }, { 24992, 13 } };
        private Dictionary<string, int> _textnormDict = new Dictionary<string, int>() { { "withitn", 14 }, { "woitn", 15 } };
        private Dictionary<int, int> _textnormIntDict = new Dictionary<int, int>() { { 25016, 14 }, { 25017, 15 } };

        public OfflineProjOfSenseVoiceSmall(OfflineModel offlineModel)
        {
            _offlineModel = offlineModel;
            _modelSession = offlineModel.ModelSession;
            _embedSession = offlineModel.EmbedSession;
            _useITN = offlineModel.UseITN;
            _tokenizer = AutoTokenizer.Create(type: TokenizerType.Textoken, vocabFilePath: offlineModel.TokensFilePath);
        }
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }
        public ITokenizer Tokenizer { get => _tokenizer; set => _tokenizer = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public bool IsPaddingSpeech { get => _isPaddingSpeech; set => _isPaddingSpeech = value; }
        public bool IsResizeAudioDuration { get => _isResizeAudioDuration; set => _isResizeAudioDuration = value; }

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
        public float[] EmbedProj(Int64[] x, int speechSize = 0)
        {
            float[] y = new float[0];
            var inputMeta = _embedSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { 1, x.Length };
                    var tensor = new DenseTensor<Int64>(x, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? results = null;
            try
            {
                results = _embedSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                    y = logits_tensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Sensevoice embed model infer failed", ex.InnerException);
            }
            return y;
        }

        public ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            //
            string languageValue = "ja";
            int languageId = 0;
            if (_lidDict.ContainsKey(languageValue))
            {
                //languageId = _lidDict.GetValueOrDefault(languageValue);
                _lidDict.TryGetValue(languageValue,out languageId);
            }
            string textnormValue = "withitn";
            if (!_useITN)
            {
                textnormValue = "woitn";
            }
            int textnormId = 15;
            if (_textnormDict.ContainsKey(textnormValue))
            {
                //textnormId = _textnormDict.GetValueOrDefault(textnormValue);
                _textnormDict.TryGetValue(textnormValue, out languageId);
            }
            var inputMeta = _modelSession.InputMetadata;
            if (!inputMeta.ContainsKey("language") && !inputMeta.ContainsKey("textnorm"))
            {
                List<OfflineInputEntity> offlineInputEntities = new List<OfflineInputEntity>();
                foreach (OfflineInputEntity offlineInputEntity in modelInputs)
                {
                    float[]? speech = offlineInputEntity.Speech;
                    if (speech != null)
                    {
                        float[] language_query = EmbedProj(new Int64[] { languageId });
                        float[] textnorm_query = EmbedProj(new long[] { textnormId });
                        //
                        float[] tempSpeech = new float[speech.Length + 560];
                        Array.Copy(textnorm_query, 0, tempSpeech, 0, textnorm_query.Length);
                        Array.Copy(speech, 0, tempSpeech, textnorm_query.Length, speech.Length);
                        speech = tempSpeech;
                        //
                        float[] event_emo_query = EmbedProj(new Int64[] { 1, 2 });
                        float[] input_query = new float[language_query.Length + event_emo_query.Length];
                        Array.Copy(language_query, 0, input_query, 0, language_query.Length);
                        Array.Copy(event_emo_query, 0, input_query, language_query.Length, event_emo_query.Length);
                        //
                        float[] tempSpeech2 = new float[speech.Length + input_query.Length];
                        Array.Copy(input_query, 0, tempSpeech2, 0, input_query.Length);
                        Array.Copy(speech, 0, tempSpeech2, input_query.Length, speech.Length);
                        speech = tempSpeech2;
                    }
                    offlineInputEntity.Speech = speech;
                    offlineInputEntity.SpeechLength = speech.Length;
                    offlineInputEntities.Add(offlineInputEntity);
                }
                modelInputs = offlineInputEntities;
            }
            float[] padSequence = PadHelper.PadSequence(modelInputs);
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
                if (name == "language")
                {
                    int[] language = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        language[i] = languageId;
                    }
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(language, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "textnorm")
                {
                    int[] textnorm = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        textnorm[i] = textnormId;
                    }
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(textnorm, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
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
                throw new Exception("Sensevoice model infer failed", ex);
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
        ~OfflineProjOfSenseVoiceSmall()
        {
            Dispose(_disposed);
        }
    }
}
