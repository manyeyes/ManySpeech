using System.Text;
using System.Text.RegularExpressions;

namespace ManySpeech.MoonshineAsr.Utils
{
    internal class ResultHelper
    {
        internal static string CheckText(string text)
        {
            Regex r = new Regex(@"\<(\w+)\>");
            var matches = r.Matches(text);
            int mIndex = -1;
            List<string> hexsList = new List<string>();
            List<string> strsList = new List<string>();
            StringBuilder hexSB = new StringBuilder();
            foreach (var m in matches.Cast<Match>().ToArray())
            {
                if (mIndex == -1)
                {
                    hexSB.Append(m.Groups[0].ToString());
                }
                else
                {
                    if (m.Index - mIndex == 6)
                    {
                        hexSB.Append(m.Groups[0].ToString());
                    }
                    else
                    {
                        hexsList.Add(hexSB.ToString());
                        strsList.Add(hexSB.ToString().Replace("<0x", "").Replace(">", ""));
                        hexSB = new StringBuilder();
                        hexSB.Append(m.Groups[0].ToString());
                    }
                }
                if (m == matches.Cast<Match>().Last())
                {
                    hexsList.Add(hexSB.ToString());
                    strsList.Add(hexSB.ToString().Replace("<0x", "").Replace(">", ""));
                }
                mIndex = m.Index;
            }
#if NET6_0_OR_GREATER || NETCOREAPP3_1_OR_GREATER
            // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
            foreach (var item in hexsList.Zip<string, string>(strsList))
            {
                text = text.Replace(item.First, HexToStr(item.Second));
            }
#else
            // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
            for (int i = 0; i < hexsList.Count && i < strsList.Count; i++)
            {
                text = text.Replace(hexsList[i], HexToStr(strsList[i]));
            }
#endif
            return text;
        }

        /// <summary>
        /// 从16进制转换成汉字
        /// </summary>
        /// <param name="hex"></param>
        /// <returns></returns>
        internal static string HexToStr(string hex)
        {
            if (hex == null)
                throw new ArgumentNullException("hex");
            if (hex.Length % 2 != 0)
            {
                hex += "20";//空格
            }
            // 需要将 hex 转换成 byte 数组。
            byte[] bytes = new byte[hex.Length / 2];
            for (int i = 0; i < bytes.Length; i++)
            {
                try
                {
                    // 每两个字符是一个 byte。
                    bytes[i] = byte.Parse(hex.Substring(i * 2, 2),
                        System.Globalization.NumberStyles.HexNumber);
                }
                catch
                {
                    throw new ArgumentException("hex is not a valid hex number!", "hex");
                }
            }
            string str = Encoding.GetEncoding("utf-8").GetString(bytes);
            return str;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        internal static bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }
    }
}
