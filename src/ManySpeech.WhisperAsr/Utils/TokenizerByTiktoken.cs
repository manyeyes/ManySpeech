using System.Diagnostics;
using System.Text.RegularExpressions;
using Tiktoken;

namespace ManySpeech.WhisperAsr.Utils
{
    internal class Tokenizer : IDisposable
    {
        private bool _disposed;
        private GptEncoding _gptEncoding;
        private int _eot;
        private int _transcribe;
        private int _translate;
        private int _sot;
        private int _sotLm;
        private int _sotPrev;
        private int _noSpeech;
        private int _noTimestamps;
        private int _timestampBegin;
        private int _languageToken;
        private List<int> _allLanguageTokens = new List<int>();
        private List<string> _allLanguageCodes = new List<string>();
        private List<int> _sotSequence = new List<int>();
        private List<int> _sotSequenceIncludingNotimestamps = new List<int>();
        private List<int> _nonSpeechTokens = new List<int>();
        private string? _language;
        private string? _task;
        private int _numLanguages;

        public int Eot { get => _eot; set => _eot = value; }
        public int Transcribe { get => _transcribe; set => _transcribe = value; }
        public int Translate { get => _translate; set => _translate = value; }
        public int Sot { get => _sot; set => _sot = value; }
        public int SotLm { get => _sotLm; set => _sotLm = value; }
        public int SotPrev { get => _sotPrev; set => _sotPrev = value; }
        public int NoSpeech { get => _noSpeech; set => _noSpeech = value; }
        public int NoTimestamps { get => _noTimestamps; set => _noTimestamps = value; }
        public int TimestampBegin { get => _timestampBegin; set => _timestampBegin = value; }
        public int LanguageToken { get => _languageToken; set => _languageToken = value; }
        public List<int> AllLanguageTokens { get => _allLanguageTokens; set => _allLanguageTokens = value; }
        public List<string> AllLanguageCodes { get => _allLanguageCodes; set => _allLanguageCodes = value; }
        public List<int> SotSequence { get => _sotSequence; set => _sotSequence = value; }
        public List<int> SotSequenceIncludingNotimestamps { get => _sotSequenceIncludingNotimestamps; set => _sotSequenceIncludingNotimestamps = value; }
        public List<int> NonSpeechTokens { get => _nonSpeechTokens; set => _nonSpeechTokens = value; }
        public string Language { get => _language; set => _language = value; }
        public string Task { get => _task; set => _task = value; }

        public Tokenizer(GptEncoding encoding, int numLanguages = 99, string? language = null, string? task = null, List<int> sotSequence = null, Dictionary<string, int>? specialTokens = default)
        {
            _numLanguages = numLanguages;
            _task = task;
            _language = language;
            _gptEncoding = encoding;
            _eot = encoding.SpecialTokenMappings.GetValueOrDefault("<|endoftext|>");
            _transcribe = encoding.SpecialTokenMappings.GetValueOrDefault("<|transcribe|>");
            _translate = encoding.SpecialTokenMappings.GetValueOrDefault("<|translate|>");
            _sot = encoding.SpecialTokenMappings.GetValueOrDefault("<|startoftranscript|>");
            _sotLm = encoding.SpecialTokenMappings.GetValueOrDefault("<|startoflm|>");
            _sotPrev = encoding.SpecialTokenMappings.GetValueOrDefault("<|startofprev|>");
            _noSpeech = encoding.SpecialTokenMappings.GetValueOrDefault("<|nospeech|>");
            _noTimestamps = encoding.SpecialTokenMappings.GetValueOrDefault("<|notimestamps|>");
            _timestampBegin = encoding.SpecialTokenMappings.GetValueOrDefault("<|0.00|>");
            _languageToken = -1;
            if (!string.IsNullOrEmpty(language))
            {
                encoding.SpecialTokenMappings.TryGetValue("<|" + language + "|>", out _languageToken);
            }
            else
            {
                Console.WriteLine("This tokenizer does not have language token configured");
            }
            if (_languageToken < 0)
            {
                Console.WriteLine(string.Format("Language {0} not found in tokenizer.", language));
            }
            _allLanguageTokens = encoding.SpecialTokenMappings.Where(x => Tiktoken.Dict.LANGUAGES.ContainsKey(Regex.Replace(x.Key, $"[{Regex.Escape("<|>")}]", string.Empty))).Select(x => x.Value).ToList();
            _allLanguageCodes = _allLanguageTokens.Select(x => Regex.Replace(decode(new List<int>() { x }), $"[{Regex.Escape("<|>")}]", string.Empty)).ToList();

            string[] langs = Tiktoken.Dict.LANGUAGES.Keys.ToArray();

            sotSequence = new List<int> { _sot };
            if (language != null)
            {
                sotSequence.Add(_sot + 1 + langs.ToList().IndexOf(language));
            }
            if (_task != null)
            {
                int taskToken = _task == "transcribe" ? _transcribe : _translate;
                sotSequence.Add(taskToken);
            }
            _sotSequence = sotSequence;
            _sotSequenceIncludingNotimestamps.AddRange(sotSequence);
            _sotSequenceIncludingNotimestamps.Add(_noTimestamps);
            _nonSpeechTokens = GetNonSpeechTokens();
        }

        public void ChangeSotSequence(string language)
        {
            string[] langs = Tiktoken.Dict.LANGUAGES.Keys.ToArray();
            List<int> sotSequence = new List<int> { _sot };
            if (language != null)
            {
                sotSequence.Add(_sot + 1 + langs.ToList().IndexOf(language));
            }
            if (_task != null)
            {
                int taskToken = _task == "transcribe" ? _transcribe : _translate;
                sotSequence.Add(taskToken);
            }
            _sotSequence = sotSequence;
            _sotSequenceIncludingNotimestamps = new List<int>(sotSequence);
            _sotSequenceIncludingNotimestamps.Add(_noTimestamps);
        }
        private List<int> GetNonSpeechTokens()
        {
            string symbolString = "\" # ( ) * + / : ; < = > @ [ \\ ] ^ _ ` { | } ~ 「 」 『 』 ";
            symbolString += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪";
            List<string> symbols = symbolString.Split().ToList();
            // symbols that may be a single token or multiple tokens depending on the tokenizer.
            // In case they're multiple tokens, suppress the first token, which is safe because:
            // These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
            // in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
            HashSet<char> miscellaneous = new HashSet<char> { '♩', '♪', '♫', '♬', '♭', '♮', '♯' };
            // 断言检查：确保miscellaneous集合中的所有字符的Unicode码点在0x2640到0x267F之间  
            Debug.Assert(miscellaneous.All(c => 0x2640 <= (int)c && (int)c <= 0x267F),
                "Not all characters in the miscellaneous set fall within the Unicode range 0x2640 to 0x267F.");
            // allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
            List<int> result = new List<int>();
            result.Add(encode(" -")[0]);
            result.Add(encode(" '")[0]);
            symbols.AddRange(miscellaneous.Select(c => c.ToString()).ToList());
            foreach (string symbol in symbols)
            {
                foreach (List<int> tokens in new List<List<int>>() { encode(symbol.ToString()), encode(" " + symbol.ToString()) })
                {
                    if (tokens.Count == 1 || miscellaneous.Select(c => c.ToString()).ToList().Contains(symbol))
                    {
                        result.Add(tokens[0]);
                    }
                }
            }
            result.Sort();
            return result;
        }


        public Tuple<string[], List<int[]>> SplitToWordTokens(List<int> tokens)
        {
            string[] langs = new string[] { "zh", "ja", "th", "lo", "my", "yue" };
            if (langs.Contains(_language))
            {
                // These languages don't typically use spaces, so it is difficult to split words
                // without morpheme analysis. Here, we instead split words at any
                // position where the tokens are decoded as valid unicode points
                return SplitTokensOnUnicode(tokens);
            }
            else
            {
                return SplitTokensOnSpaces(tokens);
            }
        }

        public Tuple<string[], List<int[]>> SplitTokensOnUnicode(List<int> tokens)
        {
            string decodedFull = DecodeWithTimestamps(tokens);
            char replacementChar = '\ufffd';

            List<string> words = new List<string>();
            List<int[]> wordTokens = new List<int[]>();
            List<int> currentTokens = new List<int>();
            int unicode_offset = 0;

            foreach (int token in tokens)
            {
                currentTokens.Add(token);
                string decoded = DecodeWithTimestamps(currentTokens);

                if (
                     decoded.Contains(replacementChar) ||
                     decodedFull[unicode_offset + decoded.IndexOf(replacementChar)] == replacementChar
                )
                {
                    words.Add(decoded);
                    wordTokens.Add(currentTokens.ToArray());
                    currentTokens = new List<int>();
                    unicode_offset += decoded.Length;
                }
            }
            return new Tuple<string[], List<int[]>>(words.ToArray(), wordTokens);
        }

        public Tuple<string[], List<int[]>> SplitTokensOnSpaces(List<int> tokens)
        {
            Tuple<string[], List<int[]>> subwords_and_subword_tokens_list = SplitTokensOnUnicode(tokens);
            List<string> words = new List<string>();
            List<int[]> wordTokens = new List<int[]>();
#if NET6_0_OR_GREATER
            foreach (var subwordAndSubwordTokens in subwords_and_subword_tokens_list.Item1.Zip<string, int[]>(subwords_and_subword_tokens_list.Item2))
            {
                string subword = subwordAndSubwordTokens.First;
                int[] subwordTokens = subwordAndSubwordTokens.Second;
#else
            for (int i = 0; i < subwords_and_subword_tokens_list.Item1.Length && i < subwords_and_subword_tokens_list.Item2.Count; i++)
            {
                string subword = subwords_and_subword_tokens_list.Item1[i];
                int[] subwordTokens = subwords_and_subword_tokens_list.Item2[i];
#endif
                bool special = subwordTokens[0] >= _eot;
                bool withSpace = subword.StartsWith(" ");
                string punc = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"; //"[\s+\.\!\/_,$%^*(+\"\']+|[▁+——！，。？：、~@#￥%……&*（）]+";
                bool punctuation = punc.Contains(subword.Trim());
                if (special || withSpace || punctuation || words.Count == 0)
                {
                    words.Add(subword);
                    wordTokens.Add(subwordTokens);
                }
                else
                {
                    string wordsLast = words.Last() + subword;
                    words.Remove(words.Last());
                    words.Add(wordsLast);
                    List<int> wordTokensLast = wordTokens.Last().ToList();
                    wordTokensLast.AddRange(subwordTokens.ToList());
                    wordTokens.Remove(wordTokens.Last());
                    wordTokens.Add(wordTokensLast.ToArray());
                }
            }
            return new Tuple<string[], List<int[]>>(words.ToArray(), wordTokens);
        }

        public List<int> encode(string text)
        {
            return _gptEncoding.Encode(text).ToList();
        }

        public string decode(List<int> tokenIds)
        {
            List<int> tokenIdsTemp = new List<int>();
            for (int i = 0; i < tokenIds.Count; i++)
            {
                int t = tokenIds[i];
                if (t >= _timestampBegin)
                {
                    //tokenIds.Remove(t);
                }
                else
                {
                    tokenIdsTemp.Add(t);
                }
            }

            return _gptEncoding.Decode(tokenIdsTemp);
        }

        public string DecodeWithTimestamps(List<int> tokenIds)
        {
            return _gptEncoding.Decode(tokenIds);
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_gptEncoding != null)
                    {
                        _gptEncoding=null;
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
        ~Tokenizer()
        {
            Dispose(_disposed);
        }
    }
}
