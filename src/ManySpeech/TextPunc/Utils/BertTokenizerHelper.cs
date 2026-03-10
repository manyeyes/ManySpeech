namespace ManySpeech.TextPunc.Utils
{
    public class BertTokenizerHelper
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly int _unkId, _clsId, _sepId, _padId;
        private readonly int _maxLength;

        public BertTokenizerHelper(string vocabPath, int maxLength = int.MaxValue)
        {
            _maxLength = maxLength;
            _vocab = File.ReadLines(vocabPath)
                         .Select((line, idx) => new { Token = line.Trim(), Id = idx })
                         .Where(x => !string.IsNullOrEmpty(x.Token))
                         .ToDictionary(x => x.Token, x => x.Id);

            if (!_vocab.TryGetValue("[UNK]", out _unkId))
                throw new Exception("词汇表缺少 [UNK]");
            if (!_vocab.TryGetValue("[CLS]", out _clsId))
                throw new Exception("词汇表缺少 [CLS]");
            if (!_vocab.TryGetValue("[SEP]", out _sepId))
                throw new Exception("词汇表缺少 [SEP]");
            if (!_vocab.TryGetValue("[PAD]", out _padId))
                throw new Exception("词汇表缺少 [PAD]");
        }

        // 分词（返回 token 字符串列表，已包含 [CLS] 和 [SEP]）
        public List<string> Tokenize(string text)
        {
            var tokens = new List<string> { "[CLS]" };
            // 简单按空格切分，对于中文可改为按字符分割（见注意事项）
            //var words = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var words = SentenceHelper.CodeMixSplitWords(text);
            foreach (var word in words)
            {
                tokens.AddRange(WordPieceTokenize(word.ToLower().Replace("▁","")));
            }
            tokens.Add("[SEP]");
            return tokens;
        }

        private List<string> WordPieceTokenize(string word)
        {
            var subTokens = new List<string>();
            var remaining = word;
            while (!string.IsNullOrEmpty(remaining))
            {
                string bestMatch = null;
                int len = remaining.Length;
                while (len > 0)
                {
                    string candidate = remaining.Substring(0, len);
                    // 除第一个子词外，其他需要加 ##
                    if (subTokens.Count > 0)
                        candidate = "##" + candidate;
                        //candidate = candidate;

                    if (_vocab.ContainsKey(candidate))
                    {
                        bestMatch = candidate;
                        break;
                    }
                    len--;
                }

                if (bestMatch == null)
                {
                    subTokens.Add("[UNK]");
                    break;
                }
                subTokens.Add(bestMatch);
                remaining = remaining.Substring(len); // 继续处理剩余部分
            }
            return subTokens;
        }

        // 将 token 列表转为 ID 列表
        public List<int> ConvertTokensToIds(List<string> tokens) =>
            tokens.Select(t => _vocab.TryGetValue(t, out int id) ? id : _unkId).ToList();

        /// <summary>
        /// 对文本进行编码
        /// </summary>
        /// <param name="text">输入文本</param>
        /// <param name="padToMaxLength">是否填充到 maxLength（默认 false，返回实际长度）</param>
        /// <returns>包含 input_ids, attention_mask, token_type_ids 的元组（均为实际长度或固定长度数组）</returns>
        public (int[] InputIds, int[] AttentionMask, int[] TokenTypeIds) Encode(string text, bool padToMaxLength = false)
        {
            // 1. 分词并转 ID
            var tokens = Tokenize(text);
            var ids = ConvertTokensToIds(tokens);

            // 2. 截断（保留 [CLS] 和尽可能多的 token，确保最后一个是 [SEP]）
            if (ids.Count > _maxLength)
            {
                ids = ids.Take(_maxLength - 1).ToList(); // 保留前 _maxLength-1 个
                ids.Add(_sepId);                          // 强制以 [SEP] 结尾
            }
            ids.RemoveAt(ids.Count()-1);
            ids.RemoveAt(0);

            // 3. 构建 attention_mask（全 1）
            var attentionMask = Enumerable.Repeat(1, ids.Count).ToArray();
            var tokenTypeIds = new int[ids.Count]; // 全 0

            // 4. 如果不填充，直接返回实际长度的数组
            if (!padToMaxLength || _maxLength == int.MaxValue)
                return (ids.ToArray(), attentionMask, tokenTypeIds);

            // 5. 填充到固定长度
            int[] paddedInputIds = new int[_maxLength];
            paddedInputIds = paddedInputIds.Select(x => x = _padId).ToArray();
            int[] paddedAttentionMask = new int[_maxLength];
            int[] paddedTokenTypeIds = new int[_maxLength];

            Array.Copy(ids.ToArray(), paddedInputIds, ids.Count);
            Array.Copy(attentionMask, paddedAttentionMask, ids.Count);
            // token_type_ids 已经是 0，不需要额外操作

            // 剩余部分保持 0（即 [PAD] 的 ID）
            return (paddedInputIds, paddedAttentionMask, paddedTokenTypeIds);
        }
    }
}