using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.Maui.Sample.Utils
{
    internal class ConvertHelper
    {
        public static IEnumerable<(int outerIndex, int innerIndex, int[] timestamp)>? Convert(List<List<int[]>> nestedList, List<int> orderIndexList)
        {
            if (nestedList == null) return null;
            if (orderIndexList.Count == 0 || orderIndexList == null)
            {
                orderIndexList = new int[nestedList.Count].ToList();
            }
            // 扁平化处理：为每个 int [] 添加对应的外层索引（从 1 开始）
            var flatItems = nestedList
            .SelectMany((innerList, outerIndex) =>
            innerList.Select((arr, innerIndex) => (
                outerIndex: outerIndex + 1, // 外层索引从 1 开始 
                innerIndex: innerIndex + 1 + orderIndexList[outerIndex], // 内层索引从 1 开始
                 timestamp: arr
            ))
            );
            return flatItems;
        }
    }
}
