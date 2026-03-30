using System.Text.RegularExpressions;

namespace PreProcessUtils
{
    public class AEDEmojiHelper
    {
        public static string ReplaceTagsWithEmojis(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }
            // 定义标签与表情包的映射关系
            var emojiMap = new System.Collections.Generic.Dictionary<string, string>
            {
                { "Laughter", "😆" },
                { "Applause", "👏" },
                { "HAPPY", "😀" },
                { "SAD", "😢" },
                { "ANGRY", "😡" },
                { "NEUTRAL", "😐" },
                { "FEARFUL", "😨" },
                { "DISGUSTED", "🤢" },
                { "SURPRISED", "😲" },
                { "Cry", "😭" },
                { "Sneeze", "👃🤧" },
                { "Cough", "🤒" },
                { "Sing", "🎤" }
            };

            string pattern = @"<\|(\w+)\|>";
            return Regex.Replace(input, pattern, match =>
            {
                string tag = match.Groups[1].Value;
                if (emojiMap.TryGetValue(tag, out string emoji))
                {
                    return emoji;
                }
                return "";
            });
        }

        public static string ReplaceTagsWithEmpty(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }
            string pattern = @"<\|.*?\|>";
            return Regex.Replace(input, pattern, match =>
            {
                return "";
            });
        }

        public static string ReplaceTagsWithEmpty2(string input, string pattern = @"<.*?>")
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }
            // 正则表达式：匹配 < 开头、> 结尾的任意字符（非贪婪匹配）
            // \< 转义匹配 <，\> 转义匹配 >，.*? 非贪婪匹配中间任意字符
            return Regex.Replace(input, pattern, "", RegexOptions.Singleline);
        }
    }
}
