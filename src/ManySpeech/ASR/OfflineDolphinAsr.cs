using ManySpeech.ASR.Model;
using ManySpeech.ASR.Utils;
using ManySpeech.SeqUnit;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;

namespace ManySpeech.ASR
{
    internal class OfflineDolphinAsr : IOffline, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private CustomMetadata _customMetadata;
        private OfflineModel _offlineModel;
        private ITokenizer _tokenizer;
        private int _sampleRate = 16000;
        private int _speechLength = 30;
        private bool _isResizeAudioDuration = false;
        private bool _isPaddingSpeech = false;
        private bool _isSampleScalingRequired = false;

        public OfflineDolphinAsr(OfflineModel offlineModel)
        {
            _offlineModel = offlineModel;
            _encoderSession = offlineModel.EncoderSession;
            _decoderSession = offlineModel.DecoderSession;
            //_customMetadata = offlineModel.CustomMetadata;
            _tokenizer = AutoTokenizer.Create(type: TokenizerType.Textoken, vocabFilePath: offlineModel.TokensFilePath);
        }
        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }
        public ITokenizer Tokenizer { get => _tokenizer; set => _tokenizer = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public bool IsPaddingSpeech { get => _isPaddingSpeech; set => _isPaddingSpeech = value; }
        public bool IsResizeAudioDuration { get => _isResizeAudioDuration; set => _isResizeAudioDuration = value; }
        public bool IsSampleScalingRequired { get => _isSampleScalingRequired; set => _isSampleScalingRequired = value; }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            int batchSize = statesList.Count;
            Debug.Assert(statesList[0].Count % 6 == 0, "when stack_states, state_list[0] is 6x");
            int fsmnLayer = statesList[0].Count;
            for (int i = 0; i < fsmnLayer; i++)
            {
                float[] statesItemTemp = new float[statesList[0][i].Length * batchSize];
                int statesItemTemp_item_length = statesList[0][i].Length;
                int statesItemTemp_item_axisnum = 512;
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
            Debug.Assert(states.Count % 6 == 0, "when stack_states, state_list[0] is 6x");
            int fsmnLayer = states.Count;
            int batchSize = states[0].Length / 512;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 512;
                    int statesItemTemp_size = 1 * 512;
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

        public List<int> GetDecoderInitTokenIds()
        {
            return new List<int> { OfflineModel.SosId };
        }

        /// <summary>
        /// 离线推理主流程（工业级健壮版）
        /// 不修改外部列表引用，仅原地修改内容
        /// </summary>
        public void Infer(
            List<OfflineInputEntity> modelInputs,
            List<List<int>> tokenIdsList,
            List<List<int[]>> timestampsList,
            List<string>? languages = null,
            List<string>? regions = null)
        {
            // 防御性校验（工业级必备）
            if (modelInputs == null) throw new ArgumentNullException(nameof(modelInputs));
            if (tokenIdsList == null) throw new ArgumentNullException(nameof(tokenIdsList));
            if (timestampsList == null) throw new ArgumentNullException(nameof(timestampsList));
            if (modelInputs.Count == 0) return;
            if (tokenIdsList.Count != modelInputs.Count)
                throw new ArgumentException("tokenIdsList 必须与 modelInputs 数量保持一致");

            int batchSize = modelInputs.Count;

            // 1. 编码器前向计算
            EncoderOutputEntity encoderOutput = EncoderProj(modelInputs);
            if (encoderOutput == null || encoderOutput.EncOut == null)
                throw new InvalidOperationException("编码器输出为空");

            // 2. 自动检测语言/地区（仅当所有序列长度=1时执行）
            bool needDetectLangRegion = true;
            foreach (var tokens in tokenIdsList)
            {
                if (tokens.Count != 1)
                {
                    needDetectLangRegion = false;
                    break;
                }
            }

            if (needDetectLangRegion)
            {
                // 只保留第一个Token（原地裁剪，无GC）
                for (int i = 0; i < batchSize; i++)
                {
                    var tokens = tokenIdsList[i];
                    if (tokens.Count > 1)
                    {
                        int keep = tokens[0];
                        tokens.Clear();
                        tokens.Add(keep);
                    }
                }

                // 检测语言
                DecoderOutputEntity langDecoderOutput = DecoderProj(encoderOutput, tokenIdsList);
                List<List<int>> langIds = DetectLanguage(langDecoderOutput.LogitsTensor);

                for (int i = 0; i < batchSize; i++)
                {
                    if (langIds[i].Count == 0) continue;
                    int langId = langIds[i][0];
                    tokenIdsList[i].Add(langId);
                    languages?.Add(Tokenizer.Decode(new[] { langId }).FirstOrDefault() ?? string.Empty);
                }

                // 检测地区
                DecoderOutputEntity regionDecoderOutput = DecoderProj(encoderOutput, tokenIdsList);
                List<List<int>> regionIds = DetectRegion(regionDecoderOutput.LogitsTensor);

                for (int i = 0; i < batchSize; i++)
                {
                    if (regionIds[i].Count == 0) continue;
                    int regionId = regionIds[i][0];
                    tokenIdsList[i].Add(regionId);
                    regions?.Add(Tokenizer.Decode(new[] { regionId }).FirstOrDefault() ?? string.Empty);
                }

                // 追加 ASR 任务标记
                for (int i = 0; i < batchSize; i++)
                {
                    tokenIdsList[i].Add(OfflineModel.AsrId);
                }
            }

            // 3. 解码参数
            const int beamSize = 1;
            const int nbest = 1;
            const int decodeMaxLen = 0;
            const float softmaxSmoothing = 1.25f;
            const float lengthPenalty = 0.6f;
            const float eosPenalty = 1.0f;

            // 计算最大解码步长
            int H = 512;
            int Ti = encoderOutput.EncOut.Count() / batchSize / H;
            int maxDecodeSteps = decodeMaxLen > 0 ? decodeMaxLen : Ti;
            if (maxDecodeSteps <= 0) maxDecodeSteps = 100; // 保底安全值

            // 4. 解码器自回归推理
            if (DecoderSession != null)
            {
                bool allEnd = false;
                for (int step = 0; step < maxDecodeSteps && !allEnd; step++)
                {
                    DecoderOutputEntity decodeOutput = DecoderProj(encoderOutput, tokenIdsList);
                    List<List<int>> newTokensBatch = DecodeAsr(decodeOutput.LogitsTensor);

                    // 原地追加Token（无GC、高性能）
                    allEnd = true;
                    for (int i = 0; i < batchSize; i++)
                    {
                        var currentTokens = tokenIdsList[i];
                        if (newTokensBatch[i].Count > 0)
                            currentTokens.AddRange(newTokensBatch[i]);

                        if (!currentTokens.Contains(OfflineModel.EosId))
                            allEnd = false;
                    }
                }
            }

            // 5. 后处理：截断 EOS 及后续内容 + 生成时间戳
            for (int i = 0; i < batchSize; i++)
            {
                var tokens = tokenIdsList[i];
                int eosIndex = tokens.IndexOf(OfflineModel.EosId);

                // 截断 EOS 之后内容
                if (eosIndex > -1)
                    tokens.RemoveRange(eosIndex, tokens.Count - eosIndex);

                // 时间戳（原地覆盖，不修改引用）
                var tsList = timestampsList[i];
                tsList.Clear();
                for (int j = 0; j < tokens.Count; j++)
                    tsList.Add(new int[2]);
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

                tokenIdsList.Add(ComputeHelper.RemoveDuplicatesAndBlank(batchTokenIds));
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

                tokenIdsList.Add(ComputeHelper.RemoveDuplicatesAndBlank(bestSequence.ToArray()));
                // Initialize timestamps
                timestampsList.Add(Enumerable.Repeat(new int[] { 0, 0 }, logitsTensor.Dimensions[1]).ToList());
            }
        }

        public EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength).ToArray();
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (!_offlineModel.ConfEntity.preprocessor_conf.use_wavfrontend)
                {
                    if (name == "speech")
                    {
                        int[] dim = new int[] { batchSize, padSequence.Length / batchSize };
                        var tensor = new DenseTensor<float>(padSequence, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                    if (name == "speech_lengths")
                    {
                        int[] dim = new int[] { batchSize };
                        Int64[] input_lengths_tensor = new Int64[batchSize];
                        input_lengths_tensor = inputLengths;
                        var tensor = new DenseTensor<Int64>(input_lengths_tensor, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                    }
                }
                else
                {
                    if (name == "feats")
                    {
                        int[] dim = new int[] { batchSize, padSequence.Length / batchSize / _offlineModel.FeatureDim, _offlineModel.FeatureDim };
                        var tensor = new DenseTensor<float>(padSequence, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }

                    if (name == "feats_lengths")
                    {
                        int[] dim = new int[] { batchSize };
                        Int64[] input_lengths_tensor = new Int64[batchSize];
                        for (int i = 0; i < batchSize; i++)
                        {
                            input_lengths_tensor[i] = padSequence.Length / batchSize / _offlineModel.FeatureDim;
                        }
                        var tensor = new DenseTensor<Int64>(input_lengths_tensor, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                    }
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.EncOut = encoderResultsArray[0].AsTensor<float>();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }

        public DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, List<List<int>> tokenidsList)
        {
            List<int[]> ys = new List<int[]>();
            ys = tokenidsList.Select(x => x.ToArray()).ToList();
            //Console.WriteLine(string.Join(",",ys[0]));
            int batchSize = ys.Count;
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            try
            {
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "ys")
                    {
                        int[] dim = new int[2] { batchSize, ys[0].Length };
                        var tensor = new DenseTensor<Int64>(ys.SelectMany(x => x.Select(x => (Int64)x)).ToArray(), dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                    }
                    if (name == "enc_out")
                    {
                        int[] dim = new int[3] { batchSize, encoderOutputEntity.EncOut.Count() / _offlineModel.ConfEntity.encoder_conf.output_size / batchSize, _offlineModel.ConfEntity.encoder_conf.output_size };
                        var tensor = new DenseTensor<float>(encoderOutputEntity.EncOut.ToArray(), dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                List<float> rescoring_score = new List<float>();
                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logits_tensor = decoderResultsArray[0].AsTensor<float>();
                    decoderOutputEntity.LogitsTensor = logits_tensor;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("DecoderProj failed", ex);
            }
            return decoderOutputEntity;
        }
        public List<List<int>> DecodeAsr(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        public List<List<int>> DetectLanguage(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if (j < _offlineModel.FirstLangId || j > _offlineModel.LastLangId)
                    {
                        continue;
                    }
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        public List<List<int>> DetectRegion(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if (j < _offlineModel.FirstRegionId || j > _offlineModel.LastRegionId)
                    {
                        continue;
                    }
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        private float ComputeAttentionScore(float[] prob, Int64[] hyp, int eos, int decode_out_len)
        {
            float score = 0.0f;
            for (int j = 0; j < hyp.Length; j++)
            {
                score += prob[j * decode_out_len + hyp[j]];
            }
            //score += prob[hyp.Length * decode_out_len + eos];
            return score;
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
        ~OfflineDolphinAsr()
        {
            Dispose(_disposed);
        }
    }
}
