// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// This file was copied and modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/byte_utils.py
// refer to : https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py
using System.Text;

namespace ManySpeech.K2TransducerAsr.Utils
{
    /// <summary>
    /// 用于处理字节相关编码和解码的工具类，功能类似原Python代码中对应功能
    /// </summary>
    public class ByteDataHelper
    {
        // 用于匹配空白字符并替换为空格的正则表达式对象
        private static readonly System.Text.RegularExpressions.Regex WHITESPACE_NORMALIZER = new System.Text.RegularExpressions.Regex(@"\s+");
        // 空格字符对应的ASCII码值为32，这里定义为常量方便使用
        private const char SPACE = (char)32;
        // 用于转义空格的特定字符，这里定义为常量，对应原Python代码中的值
        private const char SPACE_ESCAPE = (char)9601;
        // 用于表示未知字节的特定字符，对应原Python代码中的值
        private const char BPE_UNK = (char)8263;

        // 可打印的基本字符对应的ASCII码值列表，对应原Python代码中的定义
        private static readonly List<int> PRINTABLE_BASE_CHARS = new List<int>()
        {
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            288,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            298,
            299,
            300,
            301,
            302,
            303,
            304,
            305,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            316,
            317,
            318,
            321,
            322,
            323,
            324,
            325,
            326,
            327,
            328,
            330,
            331,
            332,
            333,
            334,
            335,
            336,
            337,
            338,
            339,
            340,
            341,
            342,
            343,
            344,
            345,
            346,
            347,
            348,
            349,
            350,
            351,
            352,
            353,
            354,
            355,
            356,
            357,
            358,
            359,
            360,
            361,
            362,
            363,
            364,
            365,
            366,
            367,
            368,
            369,
            370,
            371,
            372,
            373,
            374,
            375,
            376,
            377,
            378,
            379,
            380,
            381,
            382,
            384,
            385,
            386,
            387,
            388,
            389,
            390,
            391,
            392,
            393,
            394,
            395,
            396,
            397,
            398,
            399,
            400,
            401,
            402,
            403,
            404,
            405,
            406,
            407,
            408,
            409,
            410,
            411,
            412,
            413,
            414,
            415,
            416,
            417,
            418,
            419,
            420,
            421,
            422
        };

        // 字节到可打印基本字符的映射字典，用于编码过程
        private static readonly Dictionary<byte, char> BYTE_TO_BCHAR = new Dictionary<byte, char>();
        // 可打印基本字符到字节的映射字典，用于解码过程
        private static readonly Dictionary<char, byte> BCHAR_TO_BYTE = new Dictionary<char, byte>();

        static ByteDataHelper()
        {
            // 初始化字节到可打印基本字符的映射字典
            for (int i = 0; i < 256; i++)
            {
                byte b=(byte)i;
                BYTE_TO_BCHAR[b] = (char)PRINTABLE_BASE_CHARS[b];
            }
            // 初始化可打印基本字符到字节的映射字典，并将未知字符（BPE_UNK）映射到空格字节（32）
            foreach (var kvp in BYTE_TO_BCHAR)
            {
                BCHAR_TO_BYTE[kvp.Value] = kvp.Key;
            }
            BCHAR_TO_BYTE[BPE_UNK] = 32;
        }

        /// <summary>
        /// 将输入字符串进行字节编码，先规范化空白字符，再将字节转换为特定的可打印字符表示
        /// </summary>
        /// <param name="x">要编码的输入字符串</param>
        /// <returns>编码后的字符串</returns>
        public static string ByteEncode(string x)
        {
            // 将连续的空白字符替换为单个空格
            string normalized = WHITESPACE_NORMALIZER.Replace(x, SPACE.ToString());
            StringBuilder result = new StringBuilder();
            // 将规范化后的字符串转换为字节数组，再将每个字节转换为对应的可打印字符并拼接
            foreach (byte b in Encoding.UTF8.GetBytes(normalized))
            {
                result.Append(BYTE_TO_BCHAR[b]);
            }
            return result.ToString();
        }

        /// <summary>
        /// 将经过编码的字符串进行字节解码，尝试将可打印字符转换回字节并解码为原始字符串
        /// </summary>
        /// <param name="x">要解码的编码后字符串</param>
        /// <returns>解码后的字符串，如果解码失败则返回空字符串</returns>
        public static string ByteDecode(string x)
        {
            try
            {
                byte[] bytes = new byte[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    bytes[i] = BCHAR_TO_BYTE[x[i]];
                }
                return Encoding.UTF8.GetString(bytes);
            }
            catch (Exception)
            {
                return x;
            }
        }

        /// <summary>
        /// 智能字节解码方法，在常规解码失败时尝试通过动态规划寻找最佳恢复方式（尽可能多的有效字符）进行解码
        /// </summary>
        /// <param name="x">要解码的编码后字符串</param>
        /// <returns>解码后的字符串，尽量恢复出有效的内容</returns>
        public static string SmartByteDecode(string x)
        {
            string output = ByteDecode(x);
            if (output == "")
            {
                // 获取输入字符串的字节长度
                int n_bytes = x.Length;
                // 用于存储动态规划过程中的状态值（最大有效字符数）
                int[] f = new int[n_bytes + 1];
                // 用于存储动态规划过程中的回溯指针，记录最佳路径
                int[] pt = new int[n_bytes + 1];
                // 初始化动态规划的边界条件
                for (int i = 0; i <= n_bytes; i++)
                {
                    f[i] = 0;
                    pt[i] = 0;
                }
                // 动态规划计算过程，尝试不同的子串长度进行解码并更新最佳状态
                for (int i = 1; i <= n_bytes; i++)
                {
                    f[i] = f[i - 1];
                    pt[i] = i - 1;
                    for (int j = 1; j <= Math.Min(4, i); j++)
                    {
                        string subStr = x.Substring(i - j, j);
                        if (f[i - j] + 1 > f[i] && ByteDecode(subStr).Length > 0)
                        {
                            f[i] = f[i - j] + 1;
                            pt[i] = i - j;
                        }
                    }
                }
                int cur_pt = n_bytes;
                // 根据回溯指针，从后往前构建解码后的字符串，优先选择能成功解码的子串
                while (cur_pt > 0)
                {
                    if (f[cur_pt] == f[pt[cur_pt]] + 1)
                    {
                        output = ByteDecode(x.Substring(pt[cur_pt], cur_pt - pt[cur_pt])) + output;
                    }
                    cur_pt = pt[cur_pt];
                }
            }
            return output;
        }
    }
}