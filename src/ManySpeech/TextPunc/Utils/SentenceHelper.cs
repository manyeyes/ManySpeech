using System.Reflection;
using System.Text.RegularExpressions;
using YamlDotNet.Core.Tokens;

namespace ManySpeech.TextPunc.Utils
{
    /// <summary>
    /// SentenceHelper
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class SentenceHelper
    {
        // 核心修复：直接匹配需要保留的内容（中文+字母+数字+空格+缩写单引号）
        // 正则说明：
        // \u4e00-\u9fa5      匹配中文字符
        // [a-zA-Z]           匹配英文字母
        // \d                 匹配数字
        // \s                 匹配空格/换行
        // (?<=[a-zA-Z])'(?=[a-zA-Z]) 匹配字母间的单引号（缩写用，如don't中的'）
        private static readonly Regex _keepValidCharsRegex = new Regex(
            @"[\u4e00-\u9fa5a-zA-Z\d\s]|(?<=[a-zA-Z])'(?=[a-zA-Z])",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // 轻量版：保留特殊符号+缩写单引号，仅移除常见标点
        private static readonly Regex _keepCommonValidCharsRegex = new Regex(
            @"[\u4e00-\u9fa5a-zA-Z\d\s@#￥%&*_\-+=<>()\[\]{}|\\/.,;:!?，。！？；：""''""、·——【】《》“”‘’｛｝]|(?<=[a-zA-Z])'(?=[a-zA-Z])",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);
        private readonly string[]? _tokens;

        public string[]? Tokens => _tokens;

        public SentenceHelper(string tokensFilePath)
        {
            _tokens = ReadTokens(tokensFilePath);
        }
        public static string[] ReadTokens(string tokensFilePath)
        {
            string[] tokens = null;
            if (!string.IsNullOrEmpty(tokensFilePath) && tokensFilePath.IndexOf("/") < 0 && tokensFilePath.IndexOf("\\") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(tokensFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{tokensFilePath}' not found.");
                using (var reader = new StreamReader(stream))
                {
                    tokens = reader.ReadToEnd().Split('\n');//Environment.NewLine
                }
            }
            else
            {
                tokens = File.ReadAllLines(tokensFilePath);
            }
            return tokens;
        }

        /// <summary>
        /// 基础版：仅保留中文/英文/数字/空格/缩写单引号（100%保留don't/it's中的'）
        /// </summary>
        /// <param name="input">待处理文本</param>
        /// <returns>移除所有标点后的纯文本（保留缩写'）</returns>
        public static string RemoveAllPunctuation(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
            {
                return string.Empty;
            }

            // 核心修复：只提取需要保留的字符，拼接成最终文本
            MatchCollection matches = _keepValidCharsRegex.Matches(input);
            string result = string.Empty;
            foreach (Match match in matches)
            {
                result += match.Value;
            }

            return result.Trim();
        }

        /// <summary>
        /// 轻量版：保留特殊符号+缩写单引号，仅移除常见标点
        /// </summary>
        /// <param name="input">待处理文本</param>
        /// <returns>移除常见标点后的文本</returns>
        public static string RemoveCommonPunctuation(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
            {
                return string.Empty;
            }

            // 先移除常见标点，再通过保留逻辑确保缩写'不被删
            string temp = Regex.Replace(input,
                @"[，。！？；：""""、，·！￥……（）——+【】《》？：“”‘’｛｝￥（）【】｛｝、|；：”“,.!?;:()\[\]{}|\\/<>]",
                string.Empty);

            // 提取保留的字符（确保缩写'不丢失）
            MatchCollection matches = _keepValidCharsRegex.Matches(temp);
            string result = string.Empty;
            foreach (Match match in matches)
            {
                result += match.Value;
            }

            return result.Trim();
        }

        /// <summary>
        /// 自定义版：精准保留缩写单引号，移除指定标点
        /// </summary>
        /// <param name="input">待处理文本</param>
        /// <param name="punctuationsToRemove">需要移除的标点（如",.!？，。！"）</param>
        /// <returns>处理后的文本</returns>
        public static string RemoveCustomPunctuation(string input, string punctuationsToRemove)
        {
            if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(punctuationsToRemove))
            {
                return input ?? string.Empty;
            }

            // 转义自定义标点，排除单引号（避免误删缩写'）
            string escaped = Regex.Escape(punctuationsToRemove.Replace("'", ""));
            // 先移除指定标点（不包含'）
            string temp = Regex.Replace(input, $"[{escaped}]", string.Empty);

            // 确保缩写'保留
            MatchCollection matches = _keepValidCharsRegex.Matches(temp);
            string result = string.Empty;
            foreach (Match match in matches)
            {
                result += match.Value;
            }

            return result.Trim();
        }
        public static string[] CodeMixSplitWords(string text)
        {
            List<string> words = new List<string>();
            string[] segs = text.Split();
            foreach (var seg in segs)
            {
                //There is no space in seg.
                string current_word = "";
                foreach (var c in seg)
                {
                    if (c <= sbyte.MaxValue)
                    {
                        //This is an ASCII char.
                        current_word += c;
                    }
                    else
                    {
                        // This is a Chinese char.
                        if (current_word.Length > 0)
                        {
                            words.Add(current_word);
                            current_word = "";
                        }
                        words.Add(c.ToString());
                    }
                }
                if (current_word.Length > 0)
                {
                    words.Add(current_word + "▁");
                }
            }
            return words.ToArray();
        }

        public static int[] Tokens2ids(string[]? _tokens, string[]? splitText)
        {
            int[] ids = new int[splitText.Length];
            if (_tokens != null && splitText != null)
            {
                for (int i = 0; i < splitText.Length; i++)
                {
                    ids[i] = Array.IndexOf(_tokens, splitText[i].Trim('▁'));
                }
            }
            return ids;
        }

        public static List<T[]> SplitToMiniSentence<T>(T[] words, int wordLimit = 20)
        {
            List<T[]> wordsList = new List<T[]>();
            if (words.Length <= wordLimit)
            {
                wordsList.Add(words);
            }
            else
            {
                string[] sentences;
                int length = words.Length;
                int sentenceLen = (int)Math.Floor((double)(length / wordLimit));
                for (int i = 0; i < sentenceLen; i++)
                {
                    T[] vs = new T[wordLimit];
                    Array.Copy(words, i * wordLimit, vs, 0, wordLimit);
                    wordsList.Add(vs);
                }
                int tailLength = length % wordLimit;
                if (tailLength > 0)
                {
                    T[] vs = new T[tailLength];
                    Array.Copy(words, sentenceLen * wordLimit, vs, 0, tailLength);
                    wordsList.Add(vs);
                }
            }
            return wordsList;
        }
    }
}
