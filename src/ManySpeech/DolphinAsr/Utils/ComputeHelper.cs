using System.Diagnostics;
using System.IO;
// 根据不同框架引入对应命名空间
#if NETSTANDARD2_0 || NETSTANDARD2_1 || NETCOREAPP3_1 || NET461_OR_GREATER
// SharpZipLib的命名空间（仅在netstandard2.0/2.1/netcoreapp3.1下生效）
using ICSharpCode.SharpZipLib.Zip;
#else
// 系统原生ZLibStream的命名空间
using System.IO.Compression;
#endif
using System.Text;

namespace ManySpeech.DolphinAsr.Utils
{
    internal static class ComputeHelper
    {
        public static int ExactDiv(int x, int y)
        {
            Debug.Assert(x % y == 0);
            return x / y;
        }
        public static float[] SoftmaxCompute(float[] values)
        {
            if (values == null || values.Length == 0)
            {
                throw new ArgumentException("Input array must not be null or empty.");
            }
            // Calculate the maximum value in the input array for numerical stability 
            float maxVal = values.Max();
            float sum = 0.0F;
            float[] result = new float[values.Length];
            // Apply the Softmax to each value 
            for (int i = 0; i < values.Length; i++)
            {
                float e = (float)Math.Exp(values[i] - maxVal); // Prevent overflow  
                sum += e;
                result[i] = e;
            }
            // Normalize  
            for (int i = 0; i < values.Length; i++)
            {
                result[i] /= sum;
            }
            return result;
        }

        public static float[] LogCompute(float[] values)
        {
            values = values.Select(x => (float)Math.Log(x)).ToArray();
            return values;
        }

        public static float LogSumExp(float[] arr, int dim = -1)
        {
            // For one-dimensional arrays, dim is always 0 because there's only one dimension.
            // However, to maintain compatibility with higher-dimensional array processing, we accept the dim parameter and validate it.
            if (dim != -1 && dim != 0)
            {
                throw new ArgumentException("dim must be -1 or 0 for a one-dimensional array");
            }
            // Find the maximum value to avoid numerical overflow 
            double maxVal = arr.Max();
            // Compute the sum of exp(arr[i] - maxVal)
            double sumExp = arr.Sum(x => Math.Exp(x - maxVal));
            // Compute the final log sum
            double logSumExp = Math.Log(sumExp) + maxVal;
            return (float)logSumExp;
        }

        public static float CompressionRatio(string text)
        {
            float compressionRatio = 0f;
            string originalString = text;
            byte[] originalBytes = Encoding.UTF8.GetBytes(originalString);

            using (var memoryStream = new MemoryStream())
            {
#if NETSTANDARD2_0 || NETSTANDARD2_1 || NETCOREAPP3_1 || NET461_OR_GREATER
                // SharpZipLib（only in netstandard2.0/2.1 take effect）
                using (var zlibStream = new ZipInputStream(memoryStream, StringCodec.Default))
#else
                // 系统原生ZLibStream的命名空间（在netcoreapp3.1等框架下生效）
                using (var zlibStream = new ZLibStream(memoryStream, CompressionLevel.Fastest))
#endif
                {
                    zlibStream.Write(originalBytes, 0, originalBytes.Length);
                    zlibStream.Flush();
                }
                byte[] compressedBytes = memoryStream.ToArray(); // Get the compressed byte array
                compressionRatio = (float)originalBytes.Length / compressedBytes.Length;
                // If decompression is needed, ZlibStream can be used again with CompressionMode set to Decompress.
            }
            return compressionRatio;
        }
        /// <summary>
        /// 跳过序列的最后一个元素（兼容 .NET Standard 2.0）
        /// </summary>
        /// <typeparam name="T">序列元素类型</typeparam>
        /// <param name="source">源序列</param>
        /// <returns>排除最后一个元素的新序列</returns>
        public static IEnumerable<T> SkipLastOne<T>(this IList<T> source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            for (int i = 0; i < source.Count - 1; i++)
            {
                yield return source[i];
            }
        }

        /// <summary>
        /// CTC 对齐
        /// </summary>
        /// <param name="logProbs">每帧的对数概率数组（T帧，每帧为各token的概率）</param>
        /// <param name="targets">目标token序列</param>
        /// <param name="blank">空白token值</param>
        /// <returns>每帧对齐的token序列</returns>
        public static int[] ForcedAlign(List<float[]> logProbs, int[] targets, int blank)
        {
            int T = logProbs.Count;
            if (T == 0) return Array.Empty<int>();
            int S = targets.Length;

            // 必要帧数至少为扩展序列长度 2S+1
            int minFrames = 2 * S + 1;
            if (T < minFrames)
                throw new ArgumentException($"帧数不足：需要至少 {minFrames} 帧，实际只有 {T} 帧。");

            // 扩展目标序列：blank + t1 + blank + t2 + ... + blank
            int[] extTargets = new int[2 * S + 1];
            for (int i = 0; i < 2 * S + 1; i++)
                extTargets[i] = (i % 2 == 0) ? blank : targets[i / 2];
            int L = extTargets.Length;

            float[,] dp = new float[T, L];
            int[,] backtrack = new int[T, L];

            // 初始化第0帧：允许从空白或第一个token开始
            for (int s = 0; s < L; s++)
                dp[0, s] = float.NegativeInfinity;
            dp[0, 0] = logProbs[0][blank];
            if (L > 1) dp[0, 1] = logProbs[0][targets[0]];
            backtrack[0, 0] = backtrack[0, 1] = -1;

            // 递推
            for (int t = 1; t < T; t++)
            {
                float[] frame = logProbs[t];
                for (int s = 0; s < L; s++)
                {
                    int currentLabel = extTargets[s];
                    List<int> prevCandidates = new List<int> { s };          // 停留在同一位置
                    if (s > 0) prevCandidates.Add(s - 1);                   // 从左邻位置来

                    // 只有当前是非空白时才允许从 s-2 来（跳过中间的空白）
                    if (s > 1 && extTargets[s] != blank && extTargets[s - 2] != extTargets[s])
                    {
                        prevCandidates.Add(s - 2);
                    }

                    float maxProb = float.NegativeInfinity;
                    int bestPrev = -1;
                    foreach (int prev in prevCandidates)
                    {
                        if (dp[t - 1, prev] > maxProb)
                        {
                            maxProb = dp[t - 1, prev];
                            bestPrev = prev;
                        }
                    }

                    if (maxProb == float.NegativeInfinity)
                    {
                        dp[t, s] = float.NegativeInfinity;
                        backtrack[t, s] = -1;
                    }
                    else
                    {
                        dp[t, s] = maxProb + frame[currentLabel];
                        backtrack[t, s] = bestPrev;
                    }
                }
            }

            // 终止：选择最后两个位置中概率较大的
            int finalPos = L > 1 ? (dp[T - 1, L - 1] > dp[T - 1, L - 2] ? L - 1 : L - 2) : 0;
            if (dp[T - 1, finalPos] == float.NegativeInfinity)
                throw new Exception("无法找到有效的对齐路径");

            // 回溯得到对齐路径
            int[] alignment = new int[T];
            int curPos = finalPos;
            for (int t = T - 1; t >= 0; t--)
            {
                if (curPos == -1)
                    throw new Exception("回溯过程中遇到无效指针");
                alignment[t] = extTargets[curPos];
                if (t > 0)
                    curPos = backtrack[t, curPos];
            }

            // 验证对齐结果：按空白分隔提取 token 序列（与时间戳逻辑一致）
            List<int> decodedTokens = new List<int>();
            int prevToken = blank;
            foreach (int token in alignment)
            {
                if (token != blank)
                {
                    if (token != prevToken)
                    {
                        decodedTokens.Add(token);
                        prevToken = token;
                    }
                }
                else
                {
                    prevToken = blank;
                }
            }
            if (decodedTokens.Count != S)
            {
                Console.WriteLine($"警告：对齐解码后 token 数 {decodedTokens.Count} 与目标数 {S} 不符！");
                Console.WriteLine("目标序列: " + string.Join(", ", targets));
                Console.WriteLine("解码序列: " + string.Join(", ", decodedTokens));
                foreach (int tgt in targets)
                {
                    if (!decodedTokens.Contains(tgt))
                        Console.WriteLine($"目标 {tgt} 未出现在解码序列中！");
                }
                throw new Exception("对齐结果与目标序列不一致");
            }

            return alignment;
        }
    }
}
