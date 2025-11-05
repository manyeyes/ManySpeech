using System;
using System.Collections.Generic;
using System.Text;

namespace ManySpeech.SpeechProcessing
{
#if NET461_OR_GREATER || NETSTANDARD2_0
    public static class EnumerableExtensions
    {
        /// <summary>
        /// 跳过序列的最后 n 个元素
        /// </summary>
        /// <typeparam name="T">序列元素类型</typeparam>
        /// <param name="source">源序列</param>
        /// <param name="count">要跳过的最后元素数量</param>
        /// <returns>排除最后 n 个元素的新序列</returns>
        public static IEnumerable<T> SkipLast<T>(this IEnumerable<T> source, int count)
        {
            // 边界检查：源序列不能为 null
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            // 当 count <= 0 时，返回所有元素（通过 yield 逐个返回）
            if (count <= 0)
            {
                foreach (var item in source)
                {
                    yield return item; // 逐个返回源序列元素
                }
                yield break; // 结束迭代
            }

            // 当 count > 0 时，使用队列缓存元素
            var queue = new Queue<T>(count + 1); // 容量设为 count + 1，减少扩容

            foreach (var item in source)
            {
                queue.Enqueue(item);
                if (queue.Count > count)
                {
                    yield return queue.Dequeue(); // 当队列元素超过 count 时，输出最早的元素
                }
            }

            // 若源序列长度 <= count，循环结束后队列中元素不输出（即返回空序列）
        }
    }
#endif
}
