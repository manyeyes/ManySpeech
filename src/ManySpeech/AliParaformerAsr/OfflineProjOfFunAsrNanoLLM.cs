// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.AliParaformerAsr.Utils;
using ManySpeech.SeqUnit;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ManySpeech.AliParaformerAsr
{
    internal class OfflineProjOfFunAsrNanoLLM : IOfflineProj, IDisposable
    {
        private bool _disposed;

        private readonly InferenceSession _encoderSession;
        private readonly InferenceSession _adaptorSession;
        private readonly InferenceSession _decoderSession;
        private readonly InferenceSession _embedSession;
        private readonly OfflineModel _offlineModel;
        private readonly ITokenizer _tokenizer;
        private int _sampleRate = 16000;
        private int _speechLength = 30;
        private bool _isResizeAudioDuration = true;
        private bool _isPaddingSpeech = false;
        private bool _useITN = false;

        // 模型参数
        private const int _hiddenDim = 1024;
        private const int _numLayers = 28;
        private const int _numHeads = 8;
        private const int _headDim = 128;
        private const int _vocabSize = 151936;
        private const int _maxNewTokens = 200;
        private const float _repeatPenalty = 1.0f;
        private static readonly HashSet<int> _stopTokens = new HashSet<int> { 151643, 151645 };

        // 预计算提示词嵌入
        private readonly float[] _systemEmbed;
        private readonly float[] _userPrefixEmbed;
        private readonly float[] _assistantPrefixEmbed;

        public OfflineProjOfFunAsrNanoLLM(OfflineModel offlineModel)
        {
            _offlineModel = offlineModel;
            _encoderSession = offlineModel.EncoderSession ?? throw new ArgumentNullException(nameof(offlineModel.EncoderSession));
            _decoderSession = offlineModel.DecoderSession ?? throw new ArgumentNullException(nameof(offlineModel.DecoderSession));
            _embedSession = offlineModel.EmbedSession ?? throw new ArgumentNullException(nameof(offlineModel.EmbedSession));
            _adaptorSession = offlineModel.AdaptorSession ?? null;
            _useITN = offlineModel.UseITN;

            // 初始化 tokenizer
            _tokenizer = AutoTokenizer.Create(type: TokenizerType.Tiktoken,
                                              vocabFilePath: offlineModel.TokensFilePath,
                                              encodingName: "qwen3");

            // 预计算系统提示嵌入
            _systemEmbed = EmbedPrompt("<|im_start|>system\n\nYou are a helpful assistant.<|im_end|>\n");
            _userPrefixEmbed = EmbedPrompt("<|im_start|>user\n");
            _assistantPrefixEmbed = EmbedPrompt("<|im_end|>\n<|im_start|>assistant\n");
        }

        public OfflineModel OfflineModel => _offlineModel;
        public ITokenizer Tokenizer => _tokenizer;

        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public bool IsPaddingSpeech { get => _isPaddingSpeech; set => _isPaddingSpeech = value; }
        public bool IsResizeAudioDuration { get => _isResizeAudioDuration; set => _isResizeAudioDuration = value; }

        /// <summary>
        /// 将文本转换为嵌入（扁平化数组，形状 [1, seq_len, hidden_dim]）
        /// </summary>
        private float[] EmbedPrompt(string text)
        {
            var tokenIds = _tokenizer.Encode(text, isAllowSpecial: true);
            return EmbedProj(tokenIds);
        }

        private float[] EmbedProj(int[] tokenIds)
        {
            var inputTensor = new DenseTensor<long>(new[] { 1, tokenIds.Length });
            for (int i = 0; i < tokenIds.Length; i++)
                inputTensor[0, i] = tokenIds[i];

            string embedInputName = _embedSession.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(embedInputName, inputTensor)
            };

            using (var results = _embedSession.Run(inputs))
            {
                var outputTensor = results.First().AsTensor<float>();
                return outputTensor.ToArray(); // [1, seq_len, hidden_dim]
            }
        }

        /// <summary>
        /// 编码音频特征为嵌入（支持 batch）
        /// </summary>
        private EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession?.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    // 计算 shape: [batch, time, 560]
                    int time = padSequence.Length / 560 / batchSize;
                    int[] dim = new int[] { batchSize, time, 560 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    Int64[] speech_lengths = new Int64[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / 560 / batchSize;
                    }
                    var tensor = new DenseTensor<Int64>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            EncoderOutputEntity encoderOutputEntity = new EncoderOutputEntity();
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _encoderSession.Run(container);

                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    encoderOutputEntity.EncOut = resultsArray[0].AsTensor<float>();
                    encoderOutputEntity.EncOutLens = resultsArray[1].AsEnumerable<Int64>().ToArray();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutputEntity;
        }

        private EncoderOutputEntity AdaptorProj(EncoderOutputEntity encoderOutputEntity)
        {
            var inputMeta = _adaptorSession?.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "encoder_out")
                {
                    int[] dim = encoderOutputEntity.EncOut.Dimensions.ToArray();
                    var tensor = new DenseTensor<float>(encoderOutputEntity.EncOut.ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "encoder_out_lens")
                {
                    int[] dim = new int[] { encoderOutputEntity.EncOutLens.Length };
                    Int64[] encoder_out_lens = encoderOutputEntity.EncOutLens;
                    var tensor = new DenseTensor<Int64>(encoder_out_lens, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _adaptorSession.Run(container);

                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    encoderOutputEntity.EncOut = resultsArray[0].AsTensor<float>();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("AdaptorProj failed", ex);
            }
            return encoderOutputEntity;
        }

        /// <summary>
        /// 批量拼接嵌入，并返回 padding 后的结果
        /// </summary>
        private (float[] initHidden, int[] initLens) ConcatBatchEmbeddings(
            List<float[]> batchAudioEmbed,
            float[] systemEmbed,
            float[] userPrefixEmbed,
            float[] promptEmbed,
            float[] assistantPrefixEmbed)
        {
            int batchSize = batchAudioEmbed.Count;
            int systemLen = systemEmbed.Length / _hiddenDim;
            int userLen = userPrefixEmbed.Length / _hiddenDim;
            int promptLen = promptEmbed.Length / _hiddenDim;
            int assistantLen = assistantPrefixEmbed.Length / _hiddenDim;

            int fixedLen = systemLen + userLen + promptLen + assistantLen;
            int[] audioLens = batchAudioEmbed.Select(emb => emb.Length / _hiddenDim).ToArray();
            int[] totalLens = audioLens.Select(l => fixedLen + l).ToArray();
            int maxTotalLen = totalLens.Max();

            var result = new float[batchSize * maxTotalLen * _hiddenDim];
            for (int b = 0; b < batchSize; b++)
            {
                float[] audioEmb = batchAudioEmbed[b];
                int audioLen = audioLens[b];

                int offset = 0;
                // system
                Array.Copy(systemEmbed, 0, result, b * maxTotalLen * _hiddenDim + offset, systemEmbed.Length);
                offset += systemEmbed.Length;
                // user
                Array.Copy(userPrefixEmbed, 0, result, b * maxTotalLen * _hiddenDim + offset, userPrefixEmbed.Length);
                offset += userPrefixEmbed.Length;
                // prompt
                Array.Copy(promptEmbed, 0, result, b * maxTotalLen * _hiddenDim + offset, promptEmbed.Length);
                offset += promptEmbed.Length;
                // audio
                Array.Copy(audioEmb, 0, result, b * maxTotalLen * _hiddenDim + offset, audioEmb.Length);
                offset += audioEmb.Length;
                // assistant
                Array.Copy(assistantPrefixEmbed, 0, result, b * maxTotalLen * _hiddenDim + offset, assistantPrefixEmbed.Length);
            }

            return (result, totalLens);
        }

        /// <summary>
        /// 批量自回归解码
        /// </summary>
        private List<List<int>> DecodeBatch(float[] initHidden, int[] initLens)
        {
            int batchSize = initLens.Length;
            int maxSeqLen = initHidden.Length / batchSize / _hiddenDim;

            var outputMetadata = _decoderSession.OutputMetadata;
            var outputNames = outputMetadata.Keys.ToList();
            int expectedOutputs = _numLayers * 2 + 2;
            if (outputNames.Count != expectedOutputs)
                throw new InvalidOperationException($"Expected {expectedOutputs} outputs, but found {outputNames.Count}");

            var keyOutputIndices = Enumerable.Range(0, _numLayers).ToList();
            var valueOutputIndices = Enumerable.Range(_numLayers, _numLayers).ToList();
            int logitsIndex = _numLayers * 2;
            int kvSeqLenIndex = _numLayers * 2 + 1;

            var inputs = new Dictionary<string, NamedOnnxValue>();

            var inputMetadata = _decoderSession.InputMetadata;
            for (int i = 0; i < _numLayers; i++)
            {
                string keyName = $"in_key_{i}";
                string valueName = $"in_value_{i}";
                var keyShape = inputMetadata[keyName].Dimensions;
                var valueShape = inputMetadata[valueName].Dimensions;

                var keyDims = keyShape.Select(d => d == -1 ? 0 : d).ToArray();
                keyDims[0] = batchSize;
                var valueDims = valueShape.Select(d => d == -1 ? 0 : d).ToArray();
                valueDims[0] = batchSize;

                var keyTensor = new DenseTensor<float>(keyDims);
                var valueTensor = new DenseTensor<float>(valueDims);
                inputs[keyName] = NamedOnnxValue.CreateFromTensor(keyName, keyTensor);
                inputs[valueName] = NamedOnnxValue.CreateFromTensor(valueName, valueTensor);
            }

            var hiddenTensor = new DenseTensor<float>(new[] { batchSize, maxSeqLen, _hiddenDim });
            for (int i = 0; i < initHidden.Length; i++)
                hiddenTensor.Buffer.Span[i] = initHidden[i];
            inputs["hidden_states"] = NamedOnnxValue.CreateFromTensor("hidden_states", hiddenTensor);

            var historyLenTensor = new DenseTensor<long>(new[] { 1 });
            for (int b = 0; b < 1; b++)
                historyLenTensor[b] = 0;
            inputs["history_len"] = NamedOnnxValue.CreateFromTensor("history_len", historyLenTensor);

            var idsLenTensor = new DenseTensor<long>(new[] { 1 });
            for (int b = 0; b < 1; b++)
                idsLenTensor[b] = initLens[b];
            inputs["ids_len"] = NamedOnnxValue.CreateFromTensor("ids_len", idsLenTensor);

            var attnMaskTensor = new DenseTensor<sbyte>(new[] { 1 });
            for (int b = 0; b < 1; b++)
                attnMaskTensor[b] = 1;
            inputs["attention_mask"] = NamedOnnxValue.CreateFromTensor("attention_mask", attnMaskTensor);

            var generated = new List<List<int>>();
            for (int b = 0; b < batchSize; b++)
                generated.Add(new List<int>());
            var penalty = new float[batchSize][];
            for (int b = 0; b < batchSize; b++)
            {
                penalty[b] = new float[_vocabSize];
                for (int i = 0; i < _vocabSize; i++)
                    penalty[b][i] = 1.0f;
            }
            try
            {
                bool[] finished = new bool[batchSize];
                int step = 0;
                while (step < _maxNewTokens && finished.Any(f => !f))
                {
                    using (var results = _decoderSession.Run(inputs.Values))
                    {
                        var resultsArray = results.ToArray();

                        var logitsTensor = resultsArray[logitsIndex].AsTensor<float>();
                        var kvSeqLenTensor = resultsArray[kvSeqLenIndex].AsTensor<long>();

                        int[] nextTokens = new int[batchSize];
                        for (int b = 0; b < batchSize; b++)
                        {
                            if (finished[b])
                            {
                                nextTokens[b] = -1;
                                continue;
                            }

                            float[] logits = new float[_vocabSize];
                            for (int i = 0; i < _vocabSize; i++)
                                logits[i] = logitsTensor[b, i];

                            int nextToken = 0;
                            float maxLogit = logits[0] * penalty[b][0];
                            for (int i = 1; i < _vocabSize; i++)
                            {
                                float val = logits[i] * penalty[b][i];
                                if (val > maxLogit)
                                {
                                    maxLogit = val;
                                    nextToken = i;
                                }
                            }

                            if (_stopTokens.Contains(nextToken))
                            {
                                finished[b] = true;
                                nextTokens[b] = -1;
                            }
                            else
                            {
                                nextTokens[b] = nextToken;
                                generated[b].Add(nextToken);
                                penalty[b][nextToken] *= _repeatPenalty;
                            }
                        }

                        if (finished.All(f => f))
                            break;

                        var newHiddenTensor = new DenseTensor<float>(new[] { batchSize, 1, _hiddenDim });
                        for (int b = 0; b < batchSize; b++)
                        {
                            if (finished[b])
                                continue;
                            int tokenId = nextTokens[b];
                            var tokenEmbed = EmbedProj(new[] { tokenId });
                            for (int i = 0; i < _hiddenDim; i++)
                                newHiddenTensor[b, 0, i] = tokenEmbed[i];
                        }
                        inputs["hidden_states"] = NamedOnnxValue.CreateFromTensor("hidden_states", newHiddenTensor);

                        var newHistoryLenTensor = new DenseTensor<long>(new[] { 1 });
                        for (int b = 0; b < 1; b++)
                            newHistoryLenTensor[b] = kvSeqLenTensor[b];
                        inputs["history_len"] = NamedOnnxValue.CreateFromTensor("history_len", newHistoryLenTensor);

                        var newIdsLenTensor = new DenseTensor<long>(new[] { 1 });
                        for (int b = 0; b < 1; b++)
                            newIdsLenTensor[b] = finished[b] ? 0 : 1;
                        inputs["ids_len"] = NamedOnnxValue.CreateFromTensor("ids_len", newIdsLenTensor);

                        var newAttnMaskTensor = new DenseTensor<sbyte>(new[] { 1 });
                        for (int b = 0; b < 1; b++)
                            newAttnMaskTensor[b] = 0;
                        inputs["attention_mask"] = NamedOnnxValue.CreateFromTensor("attention_mask", newAttnMaskTensor);

                        // 更新 KV 缓存（使用原方法：ToArray + 新张量）
                        for (int i = 0; i < _numLayers; i++)
                        {
                            var keyTensor = resultsArray[i].AsTensor<float>();
                            int[] keyDim = keyTensor.Dimensions.ToArray();
                            var keyCopy = new DenseTensor<float>(keyTensor.ToArray(), keyDim, false);
                            inputs[$"in_key_{i}"] = NamedOnnxValue.CreateFromTensor($"in_key_{i}", keyCopy);

                            var valueTensor = resultsArray[i + _numLayers].AsTensor<float>();
                            int[] valueDim = valueTensor.Dimensions.ToArray();
                            var valueCopy = new DenseTensor<float>(valueTensor.ToArray(), valueDim, false);
                            inputs[$"in_value_{i}"] = NamedOnnxValue.CreateFromTensor($"in_value_{i}", valueCopy);
                        }
                    }
                    step++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("DecodeBatch failed", ex);
            }
            return generated;
        }

        public List<int> RemoveDuplicatesAndBlank(int[] yseq, int blank_id = 0)
        {
            if (yseq == null || yseq.Length == 0)
                return new List<int>();

            int prev_token = -1;
            var decoded = new List<int>();
            foreach (int token in yseq)
            {
                if (token != prev_token && token != blank_id)
                    decoded.Add(token);
                prev_token = token;
            }
            return decoded;
        }

        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList,
                  List<List<int[]>> timestampsList, List<string>? languages = null,
                  List<string>? regions = null)
        {
            if (modelInputs == null || modelInputs.Count == 0)
                return;

            // 保存每个输入对应的结果，按原始索引存放
            var allTokenIds = new List<List<int>>(new List<int>[modelInputs.Count]);
            var allTimestamps = new List<List<int[]>>(new List<int[]>[modelInputs.Count]);

            // 为每个输入添加索引，便于分组后还原顺序
            var indexedInputs = modelInputs.Select((input, idx) => new { input, idx }).ToList();

            // 按 Language 分组（null 也作为一组）
            var groups = indexedInputs.GroupBy(x => x.input.Language);

            foreach (var group in groups)
            {
                var groupInputs = group.Select(x => x.input).ToList();
                var groupIndices = group.Select(x => x.idx).ToList();

                // 处理当前分组
                ProcessBatch(groupInputs, out var batchTokenIds, out var batchTimestamps);

                // 将结果存回对应索引位置
                for (int i = 0; i < groupIndices.Count; i++)
                {
                    int origIndex = groupIndices[i];
                    allTokenIds[origIndex] = batchTokenIds[i];
                    allTimestamps[origIndex] = batchTimestamps[i];
                }
            }
            tokenIdsList.Clear();
            timestampsList.Clear();
            tokenIdsList.AddRange(allTokenIds);
            timestampsList.AddRange(allTimestamps);
        }

        // 处理一个 batch
        private void ProcessBatch(List<OfflineInputEntity> batchInputs,
                                  out List<List<int>> batchTokenIds,
                                  out List<List<int[]>> batchTimestamps)
        {
            // 1. 热词
            string[] hotwords = new string[] { "开放时间" };
            List<string>? hotwordList = batchInputs.SelectMany(x => x.Hotwords).ToList();
            if (hotwordList != null && hotwordList.Count > 0)
            {
                hotwords = hotwordList.ToArray();
            }
            else if (_offlineModel.Hotwords?.Length > 0)
            {
                hotwords = _offlineModel.Hotwords;
            }

            string hotwordsStr = string.Join(",", hotwords);
            // 根据分组语言动态设置 lang
            string lang = batchInputs.FirstOrDefault()?.Language ?? "中文";
            string prompt = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n";
            if (!string.IsNullOrWhiteSpace(hotwordsStr))
                prompt += $"热词列表：[{hotwordsStr}]\n";
            prompt += $"语音转写成：{lang}";
            if (!_useITN)
                prompt += "，不进行文本规整";
            prompt += "：";

            var promptEmbed = EmbedPrompt(prompt);

            // 2. 编码
            var encoderOutput = EncoderProj(batchInputs);
            if (_adaptorSession != null)
            {
                encoderOutput = AdaptorProj(encoderOutput);
            }
            var audioEmbedTensor = encoderOutput.EncOut;
            int batchSize = batchInputs.Count;
            int audioTimeDim = audioEmbedTensor.Dimensions[1];
            int hiddenDim = audioEmbedTensor.Dimensions[2];

            List<float[]> batchAudioEmbed = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                float[] audioEmb = new float[audioTimeDim * hiddenDim];
                for (int t = 0; t < audioTimeDim; t++)
                    for (int d = 0; d < hiddenDim; d++)
                        audioEmb[t * hiddenDim + d] = audioEmbedTensor[b, t, d];
                batchAudioEmbed.Add(audioEmb);
            }

            var (initHidden, initLens) = ConcatBatchEmbeddings(
                batchAudioEmbed,
                _systemEmbed,
                _userPrefixEmbed,
                promptEmbed,
                _assistantPrefixEmbed
            );

            var batchGenerated = DecodeBatch(initHidden, initLens);

            // 输出
            batchTokenIds = new List<List<int>>(batchSize);
            batchTimestamps = new List<List<int[]>>(batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                batchTokenIds.Add(batchGenerated[b]);
                batchTimestamps.Add(new List<int[]>());
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _offlineModel?.Dispose();
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~OfflineProjOfFunAsrNanoLLM()
        {
            Dispose(false);
        }
    }
}