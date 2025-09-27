using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ManySpeech.AliCTTransformerPunc.Examples.Utils
{
    internal class TextHelper
    {
        /// <summary>
        /// 按行读取文本文件内容，返回字符串数组，并统计字数/单词数
        /// </summary>
        /// <param name="filePath">文本文件的完整路径</param>
        /// <param name="wordsNum">输出参数：统计的字数或单词数（根据内部逻辑）</param>
        /// <returns>包含文件所有行的string[]</returns>
        /// <exception cref="FileNotFoundException">文件不存在时抛出</exception>
        public static string[] GetFileText(string filePath, ref int wordsNum)
        {
            // 初始化统计数
            wordsNum = 0;

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException("指定的文件不存在", filePath);
            }

            List<string> lines = new List<string>();

            using (StreamReader reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    lines.Add(line);

                    // 统计逻辑：根据需求选择一种方式，注释掉另一种
                    // 方式1：统计所有非空字符（包括中文、英文、数字，排除空格和控制字符）
                    wordsNum += CountCharacters(line);

                    // 方式2：统计英文单词数（以空格分隔，忽略标点和空项）
                    // wordsNum += CountWords(line);
                }
            }

            return lines.ToArray();
        }

        public static string[] GetStrText(string str, ref int wordsNum)
        {
            // 初始化统计数
            wordsNum = 0;
            List<string> lines = new List<string>();
            // 按换行符拆分字符串（支持 \r\n、\n、\r 等多种换行格式）
            string[] splitLines = str.Replace("\\n", "\n").Replace("\\r", "\r").Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (string line in splitLines)
            {
                lines.Add(line);
                // 统计逻辑：根据需求选择一种方式，注释掉另一种
                // 方式 1：统计所有非空字符（包括中文、英文、数字，排除空格和控制字符）
                wordsNum += CountCharacters(line);
                // 方式 2：统计英文单词数（以空格分隔，忽略标点和空项）
                //wordsNum += CountWords (line);
            }
            return lines.ToArray();
        }

        /// <summary>
        /// 统计一行文本中的非空字符数（适合中文、混合文本）
        /// </summary>
        private static int CountCharacters(string line)
        {
            if (string.IsNullOrWhiteSpace(line))
                return 0;

            int count = 0;
            foreach (char c in line)
            {
                // 排除空格和控制字符（如制表符）
                if (!char.IsWhiteSpace(c) && !char.IsControl(c))
                {
                    count++;
                }
            }
            return count;
        }

        /// <summary>
        /// 统计一行文本中的英文单词数（以空格分隔，忽略标点）
        /// </summary>
        private static int CountWords(string line)
        {
            if (string.IsNullOrWhiteSpace(line))
                return 0;

            // 移除标点符号，再按空格分割
            string cleaned = Regex.Replace(line, @"[^\w\s]", "");
            string[] words = cleaned.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return words.Length;
        }

        /// <summary>
        /// 判断是否为TXT文本文件（基于编码BOM和二进制特征）
        /// </summary>
        /// <param name="filePath">文件路径</param>
        /// <returns>若判断为TXT文本文件则返回true，否则返回false（存在一定局限性）</returns>
        public static bool IsTextByHeader(string filePath)
        {
            // 检查文件是否存在
            if (!File.Exists(filePath))
                return false;

            // 读取文件前8个字节（足够判断BOM和初步二进制特征）
            byte[] headerBytes = ReadFileHeader(filePath, 8);
            if (headerBytes == null || headerBytes.Length == 0)
                return false;

            // 检查是否包含常见文本编码的BOM（存在BOM则可确定为文本文件）
            if (HasTextBom(headerBytes))
                return true;

            // 无BOM时，检查是否包含明显的二进制文件特征（如不可打印字符）
            return !ContainsBinaryFeatures(headerBytes);
        }

        /// <summary>
        /// 读取文件前n个字节
        /// </summary>
        private static byte[] ReadFileHeader(string filePath, int bytesToRead)
        {
            try
            {
                using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    int bytesRead = (int)Math.Min(fs.Length, bytesToRead);
                    byte[] buffer = new byte[bytesRead];
                    fs.Read(buffer, 0, bytesRead);
                    return buffer;
                }
            }
            catch (Exception)
            {
                // 读取失败（如文件被占用、权限不足），返回null
                return null;
            }
        }

        /// <summary>
        /// 检查是否包含文本编码的BOM（字节顺序标记）
        /// </summary>
        private static bool HasTextBom(byte[] headerBytes)
        {
            // UTF-8 BOM: EF BB BF
            if (headerBytes.Length >= 3 &&
                headerBytes[0] == 0xEF &&
                headerBytes[1] == 0xBB &&
                headerBytes[2] == 0xBF)
                return true;

            // UTF-16 Big-Endian BOM: FE FF
            if (headerBytes.Length >= 2 &&
                headerBytes[0] == 0xFE &&
                headerBytes[1] == 0xFF)
                return true;

            // UTF-16 Little-Endian BOM: FF FE
            if (headerBytes.Length >= 2 &&
                headerBytes[0] == 0xFF &&
                headerBytes[1] == 0xFE)
                return true;

            // UTF-32 Big-Endian BOM: 00 00 FE FF
            if (headerBytes.Length >= 4 &&
                headerBytes[0] == 0x00 &&
                headerBytes[1] == 0x00 &&
                headerBytes[2] == 0xFE &&
                headerBytes[3] == 0xFF)
                return true;

            // UTF-32 Little-Endian BOM: FF FE 00 00
            if (headerBytes.Length >= 4 &&
                headerBytes[0] == 0xFF &&
                headerBytes[1] == 0xFE &&
                headerBytes[2] == 0x00 &&
                headerBytes[3] == 0x00)
                return true;

            return false;
        }

        /// <summary>
        /// 检查是否包含明显的二进制文件特征（非文本特征）
        /// </summary>
        private static bool ContainsBinaryFeatures(byte[] headerBytes)
        {
            // 二进制文件通常包含大量不可打印的控制字符（除了常见的换行、制表符等）
            foreach (byte b in headerBytes)
            {
                // 允许的控制字符：\t（9）、\n（10）、\r（13）、空格（32）
                if (b < 0x20 && b != 0x09 && b != 0x0A && b != 0x0D)
                {
                    // 存在其他控制字符，可能是二进制文件
                    return true;
                }
            }
            return false;
        }
    }
}
