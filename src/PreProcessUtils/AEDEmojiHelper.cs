using System.Text.RegularExpressions;

namespace PreProcessUtils
{
    public class AEDEmojiHelper
    {
        public static string ReplaceTagsWithEmojis(string input)
        {
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
            string pattern = @"<\|.*?\|>";
            return Regex.Replace(input, pattern, match =>
            {
                return "";
            });
        }
    }
}
