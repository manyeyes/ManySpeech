// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.TextPunc.Model;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace ManySpeech.TextPunc
{
    /// <summary>
    /// CTTransformer
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class PuncRestorer : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private string[]? _punc_list;
        private string[]? _punc_en_list;
        private string[]? _tokens;
        private int _period = 0;
        private readonly IPuncProj _puncProj;
        private ITextProcessor? _textProcessor;

        public PuncRestorer(string modelFilePath, string configFilePath, string tokensFilePath, int batchSize = 1, int threadsNum = 1)
        {
            PuncModel _puncModel = new PuncModel(modelFilePath, threadsNum: threadsNum);
            ConfEntity? _confEntity = Utils.PreloadHelper.ReadJson(configFilePath);
            _punc_list = _confEntity.punc_list;
            _punc_en_list = new string[_punc_list.Length];
            Array.Copy(_punc_list, _punc_en_list, _punc_list.Length);
            _punc_en_list = _punc_en_list.Select(x => x.Replace("，", ",").Replace("？", "?").Replace("。", ".").Replace("！", "!")).ToArray();
            for (int i = 0; i < _punc_en_list.Length; i++)
            {
                if (_punc_en_list[i] == ".")
                {
                    _period = i;
                }
            }
            _punc_list = _punc_list.Select(x => x.Replace(",", "，").Replace("?", "？").Replace(".", "。").Replace("!", "！")).ToArray();
            for (int i = 0; i < _punc_list.Length; i++)
            {
                if (_punc_list[i] == "。")
                {
                    _period = i;
                }
            }

            switch (_confEntity.model.ToLower())
            {
                case "ct-transformer":
                    _puncProj = new PuncProjOfCTTransformer(_puncModel);
                    _textProcessor = new Utils.SentenceProcessor(tokensFilePath);
                    break;
                case "fireredpunc":
                    _puncProj = new PuncProjOfFireRed(_puncModel);
                    _textProcessor = new Utils.BertTokenizerProcessor(tokensFilePath);
                    break;
                default:
                    _puncProj = new PuncProjOfCTTransformer(_puncModel);
                    _textProcessor = new Utils.SentenceProcessor(tokensFilePath);
                    break;
            }
        }

        public string GetResults(string text, int splitSize = 10)
        {
            text = Utils.SentenceHelper.RemoveAllPunctuation(text);
            Console.WriteLine(string.Format("text:{0}", text));
            (string[] splitText, int[] split_text_id) = _textProcessor!.ProcessText(text);
            List<string[]> mini_sentences = Utils.SentenceHelper.SplitToMiniSentence(splitText, splitSize);
            List<int[]> mini_sentences_id = Utils.SentenceHelper.SplitToMiniSentence(split_text_id, splitSize);
            Trace.Assert(mini_sentences.Count == mini_sentences_id.Count, "There were some errors in the 'SplitToMiniSentence' method. ");
            string[] cache_sent;
            int[] cache_sent_id = new int[] { };
            List<int[]> new_mini_sentences_id = new List<int[]>();
            int cache_pop_trigger_limit = 200;

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
                PuncOutputEntity modelOutput = _puncProj.ModelProj(puncInputEntities);

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
                        punctuations = punctuations.Skip(temp_punctuations.Length).ToArray();
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
                        int[] temp_punctuations = new int[punctuations.Length];
                        Array.Copy(punctuations, 0, temp_punctuations, 0, punctuations.Length);
                        temp_punctuations[punctuations.Length - 1] = _period;
                        punctuations = temp_punctuations;
                    }
                    new_mini_sentences_id.Add(punctuations);
                }
                j++;
            }

            string text_result = AddPuncToText(splitText.ToList(), new_mini_sentences_id.SelectMany(arr => arr ?? Array.Empty<int>()).ToList());            
            return text_result;
        }

        /// <summary>
        /// 根据分词结果和预测的标点类别索引，重建带标点的文本。
        /// </summary>
        /// <param name="tokens">原始 token 列表（可能包含 BERT 子词前缀 "##"）</param>
        /// <param name="preds">每个 token 对应的标点类别索引（0 表示无标点）</param>
        /// <param name="puncMap">标点类别到标点符号的映射（例如 {1: ".", 2: ",", ...}）</param>
        /// <returns>重建后的文本</returns>
        public string AddPuncToText(List<string> tokens, List<int> preds)
        {
            if (tokens == null || preds == null || tokens.Count != preds.Count)
                throw new ArgumentException("tokens 和 preds 必须长度相等");

            var result = new StringBuilder();

            for (int i = 0; i < tokens.Count; i++)
            {
                string token = tokens[i];

                // 1. 处理 BERT 子词前缀（如 "##ing"）
                if (token.StartsWith("##"))
                    token = token.Substring(2);

                // 2. 判断是否需要在当前 token 前添加空格
                else if (i > 0)
                {
                    string prevToken = tokens[i - 1];
                    bool prevIsAlphaNum = Regex.IsMatch(prevToken, "[a-zA-Z0-9#]+");
                    bool currentIsAlphaNum = Regex.IsMatch(token, "[a-zA-Z0-9#]+");
                    bool prevHasNoPunc = preds[i - 1] == 0;   // 前一个 token 后无标点

                    if (prevIsAlphaNum && currentIsAlphaNum && prevHasNoPunc)
                        result.Append(' ');
                }
                // 3. 添加当前 token
                result.Append(token);

                // 4. 如果当前预测有标点，则添加对应的标点符号
                int p = preds[i];
                if (p != 0 && p < _punc_list.Length - 1)
                {

                    Regex looseEnglishRegex = new Regex(@"^[a-zA-Z\s\d.,!?;:()\[\]{}@#$%^&*_+\-='""\\/<>|`~]+$", RegexOptions.Compiled);
                    token = token.TrimEnd('▁');
                    if (looseEnglishRegex.IsMatch(token))
                    {
                        if (_punc_en_list[p] == "_")
                        {
                            result.Append(" ");
                        }
                        else
                        {
                            result.Append(_punc_en_list![p] + " ");
                        }
                    }
                    else
                    {
                        if (_punc_list[p] != "_")
                        {                            
                            result.Append(_punc_list[p]);
                        }
                    }

                }
            }

            // 5. 合并多余空格并 trim
            string finalText = Regex.Replace(result.ToString(), @"\s+", " ").Trim();
            return finalText.Replace("▁_", " ").Replace("▁", "").Replace("  ", " ");
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
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
        ~PuncRestorer()
        {
            Dispose(_disposed);
        }
    }
}