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

namespace ManySpeech.WhisperAsr.Utils
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
        /// Skip the last element of the sequence (compatible with.NET Standard 2.0).
        /// </summary>
        /// <typeparam name="T">The type of the elements in the sequence.</typeparam>
        /// <param name="source">The source sequence.</param>
        /// <returns>A new sequence excluding the last element.</returns> 
        public static IEnumerable<T> SkipLastOne<T>(this IList<T> source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            for (int i = 0; i < source.Count - 1; i++)
            {
                yield return source[i];
            }
        }
    }
}
