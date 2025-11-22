using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.MoonshineAsr
{
#if NET461_OR_GREATER || NETSTANDARD2_0_OR_GREATER || NETCOREAPP3_1
    public static class EnumerableExtensions
    {
        /// <summary>
        /// 自定义 MaxBy 扩展方法，兼容 net461
        /// </summary>
        /// <typeparam name="TSource">集合元素类型</typeparam>
        /// <typeparam name="TKey">排序键类型（必须实现 IComparable<TKey>）</typeparam>
        /// <param name="source">源集合</param>
        /// <param name="keySelector">键选择器</param>
        /// <returns>具有最大键值的元素</returns>
        public static TSource MaxBy<TSource, TKey>(
            this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector
        ) where TKey : IComparable<TKey>
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (keySelector == null) throw new ArgumentNullException(nameof(keySelector));

            using (var enumerator = source.GetEnumerator())
            {
                // 处理空集合
                if (!enumerator.MoveNext())
                    throw new InvalidOperationException("集合为空，无法获取 MaxBy 结果");

                // 初始化最大值元素和其键
                TSource maxElement = enumerator.Current;
                TKey maxKey = keySelector(maxElement);

                // 遍历剩余元素，更新最大值
                while (enumerator.MoveNext())
                {
                    TSource current = enumerator.Current;
                    TKey currentKey = keySelector(current);
                    if (currentKey.CompareTo(maxKey) > 0)
                    {
                        maxElement = current;
                        maxKey = currentKey;
                    }
                }

                return maxElement;
            }
        }
    }
#endif
}
