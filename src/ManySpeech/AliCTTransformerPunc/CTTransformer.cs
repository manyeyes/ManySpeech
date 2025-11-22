// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliCTTransformerPunc.Model;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Text;

namespace ManySpeech.AliCTTransformerPunc
{
    /// <summary>
    /// CTTransformer
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class CTTransformer : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _onnxSession;
        private string[]? _punc_list;
        private string[]? _punc_en_list;
        private string[]? _tokens;
        private int _period = 0;

        public CTTransformer(string modelFilePath, string configFilePath, string tokensFilePath, int batchSize = 1, int threadsNum = 1)
        {
            PuncModel puncModel = new PuncModel(modelFilePath, threadsNum: threadsNum);
            _onnxSession = puncModel.ModelSession;
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
            ConfEntity? confEntity = Utils.PreloadHelper.ReadJson(configFilePath);
            _punc_list = confEntity.punc_list;
            _punc_en_list = new string[_punc_list.Length];
            Array.Copy(_punc_list, _punc_en_list, _punc_list.Length);
            _punc_en_list = _punc_en_list.Select(x => x.Replace("，", ",").Replace("？", "?").Replace("。", ".")).ToArray();
            for (int i = 0; i < _punc_en_list.Length; i++)
            {
                if (_punc_en_list[i] == ".")
                {
                    _period = i;
                }
            }
            _punc_list = _punc_list.Select(x => x.Replace(",", "，").Replace("?", "？").Replace(".", "。")).ToArray();
            for (int i = 0; i < _punc_list.Length; i++)
            {
                if (_punc_list[i] == "。")
                {
                    _period = i;
                }
            }
        }

        public string GetResults(string text, int splitSize = 10)
        {
            string[] splitText = Utils.SentenceHelper.CodeMixSplitWords(text);
            int[] split_text_id = Utils.SentenceHelper.Tokens2ids(_tokens, splitText);
            List<string[]> mini_sentences = Utils.SentenceHelper.SplitToMiniSentence(splitText, splitSize);
            List<int[]> mini_sentences_id = Utils.SentenceHelper.SplitToMiniSentence(split_text_id, splitSize);
            Trace.Assert(mini_sentences.Count == mini_sentences_id.Count, "There were some errors in the 'SplitToMiniSentence' method. ");
            string[] cache_sent;
            int[] cache_sent_id = new int[] { };
            List<int[]> new_mini_sentences_id = new List<int[]>();
            int cache_pop_trigger_limit = 200;

            //this._logger.LogInformation("punc begin");
            int j = 0;
            foreach (int[] mini_sentence_id in mini_sentences_id)
            {
                int[] miniSentenceId;
                PuncInputEntity puncInputEntities = new PuncInputEntity();
                if (cache_sent_id.Length > 0)
                {
                    miniSentenceId = new int[cache_sent_id.Length + mini_sentence_id.Length];
                    Array.Copy(cache_sent_id, 0, miniSentenceId, 0, cache_sent_id.Length);
                    Array.Copy(mini_sentence_id, 0, miniSentenceId, cache_sent_id.Length, mini_sentence_id.Length);
                }
                else
                {
                    miniSentenceId = new int[mini_sentence_id.Length];
                    miniSentenceId = mini_sentence_id;
                }
                puncInputEntities.MiniSentenceId = miniSentenceId.Select(x => x == 0 ? -1 : x).ToArray();
                puncInputEntities.TextLengths = miniSentenceId.Length;
                PuncOutputEntity modelOutput = this.Forward(puncInputEntities);

                int[] punctuations = modelOutput.Punctuations[0];
                if (j < mini_sentences_id.Count)
                {
                    int sentenceEnd = -1;
                    int last_comma_index = -1;
                    for (int i = punctuations.Length - 2; i > 1; i--)
                    {
                        if (_punc_list[punctuations[i]] == "。" || _punc_list[punctuations[i]] == "？")
                        {
                            sentenceEnd = i;
                            break;
                        }
                        if (last_comma_index < 0 && _punc_list[punctuations[i]] == "，")
                        {
                            last_comma_index = i;
                        }
                    }
                    if (sentenceEnd < 0 && miniSentenceId.Length > cache_pop_trigger_limit && last_comma_index >= 0)
                    {
                        // The sentence it too long, cut off at a comma.
                        sentenceEnd = last_comma_index;
                        punctuations[sentenceEnd] = _period;
                    }
                    cache_sent_id = new int[miniSentenceId.Length - (sentenceEnd + 1)];
                    Array.Copy(miniSentenceId, sentenceEnd + 1, cache_sent_id, 0, cache_sent_id.Length);
                    if (sentenceEnd > 0)
                    {
                        int[] temp_punctuations = new int[sentenceEnd + 1];
                        Array.Copy(punctuations, 0, temp_punctuations, 0, temp_punctuations.Length);
                        new_mini_sentences_id.Add(temp_punctuations);
                        Array.Clear(punctuations, 0, temp_punctuations.Length);
                    }
                }
                if (j == mini_sentences_id.Count - 1)
                {
                    if (_punc_list[punctuations.Last()] == "，" || _punc_list[punctuations.Last()] == "、")
                    {
                        punctuations[punctuations.Length - 1] = _period;
                    }
                    else if (_punc_list[punctuations.Last()] != "。" && _punc_list[punctuations.Last()] != "？")
                    {
                        punctuations[punctuations.Length - 1] = _period;
                        int[] temp_punctuations = new int[punctuations.Length + 1];
                        Array.Copy(punctuations, 0, temp_punctuations, 0, punctuations.Length);
                        temp_punctuations[punctuations.Length - 1] = _period;
                        punctuations = temp_punctuations;
                    }
                    new_mini_sentences_id.Add(punctuations);
                }
                j++;
            }

            string text_result = this.Decode(new_mini_sentences_id, splitText);
            return text_result;
        }

        private string Decode(List<int[]> new_mini_sentences_id, string[] splitText)
        {
            int m = 0;
            StringBuilder sb = new StringBuilder();
            foreach (int[] sentence_id in new_mini_sentences_id)
            {
                foreach (int id in sentence_id)
                {
                    if (m < splitText.Length)
                    {
                        string word = splitText[m];
                        sb.Append(word);
                        m++;
                        if (id > 1)
                        {
                            if (word.Length > 1)
                            {
                                sb.Append(_punc_en_list[id] + " ");
                            }
                            else
                            {
                                if (word.Length > 0 && word[0] > sbyte.MaxValue)
                                {
                                    sb.Append(_punc_list[id]);
                                }
                                else
                                {
                                    sb.Append(_punc_en_list[id] + " ");
                                }
                            }

                        }
                    }
                }
            }
            return sb.ToString().Replace("▁", " ");
        }

        private PuncOutputEntity Forward(PuncInputEntity modelInput)
        {
            int BatchSize = 1;
            PuncOutputEntity modelOutput = new PuncOutputEntity();
            try
            {
                var inputMeta = _onnxSession.InputMetadata;
                var container = new List<NamedOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "inputs")
                    {
                        int[] dim = new int[] { BatchSize, modelInput.TextLengths / 1 / BatchSize };
                        var tensor = new DenseTensor<int>(modelInput.MiniSentenceId, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                    }
                    if (name == "text_lengths")
                    {
                        int[] dim = new int[] { BatchSize };
                        int[] text_lengths = new int[BatchSize];
                        for (int i = 0; i < BatchSize; i++)
                        {
                            text_lengths[i] = modelInput.TextLengths / 1 / BatchSize;
                        }
                        var tensor = new DenseTensor<int>(text_lengths, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                    }
                }
                IReadOnlyCollection<string> outputNames = new List<string>();
                //outputNames.Append("logits");
                //outputNames.Append("token_num");
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;

                results = _onnxSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    modelOutput.Logits = resultsArray[0].AsEnumerable<float>().ToArray();
                    Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                    List<int[]> token_nums = new List<int[]> { };

                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        int[] item = new int[logits_tensor.Dimensions[1]];
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                        }
                        token_nums.Add(item);
                    }
                    modelOutput.Punctuations = token_nums;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Automatic punctuation failed", ex);
            }
            return modelOutput;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onnxSession != null)
                    {
                        _onnxSession?.Dispose();
                    }
                    if (_punc_list != null)
                    {
                        _punc_list = null;
                    }
                    if (_punc_en_list != null)
                    {
                        _punc_en_list = null;
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
        ~CTTransformer()
        {
            Dispose(_disposed);
        }
    }
}