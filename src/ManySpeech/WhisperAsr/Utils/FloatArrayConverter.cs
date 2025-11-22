using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
// 根据不同框架引入必要的命名空间
#if NETCOREAPP3_1 || NET5_0_OR_GREATER || NETSTANDARD2_1
using System; // 这些框架原生支持Half类型
#elif NET461 || NET472 || NET48 || NETSTANDARD2_0
using System.Numerics; // 需要通过NuGet安装System.Numerics.Vectors
#endif

namespace ManySpeech.WhisperAsr.Utils
{
    public class FloatArrayConverter
    {
        [StructLayout(LayoutKind.Explicit)]
        public struct FloatConverter
        {
            [FieldOffset(0)]
            public float Float;
            [FieldOffset(0)]
            public ushort ShortH;
            [FieldOffset(2)]
            public ushort ShortL;
        }

        public static unsafe Float16[] ConvertFloat32ToFloat16(float[] floatArray)
        {
            if (floatArray == null)
                throw new ArgumentNullException(nameof(floatArray));

            int length = floatArray.Length;
            if (length % 2 != 0)
                throw new ArgumentException("The length of the float array must be even.", nameof(floatArray));

            Float16[] halfArray = new Float16[length];
            FloatConverter converter = new FloatConverter();

            for (int i = 0; i < length; i += 2)
            {
                converter.Float = floatArray[i];
                halfArray[i] = (Float16)converter.ShortH;

                converter.Float = floatArray[i + 1];
                halfArray[i + 1] = (Float16)converter.ShortL;
            }

            return halfArray;
        }
        /// <summary>
        /// 将float数组转换为Float16数组（跨框架兼容版）
        /// </summary>
        public static Float16[] FloatArrayToHalfArray(float[] floatArray)
        {
            if (floatArray == null)
                throw new ArgumentNullException(nameof(floatArray));

#if NET5_0_OR_GREATER
            // 支持原生Half类型的框架
            Half[] halfArray = new Half[floatArray.Length];
            for (int i = 0; i < floatArray.Length; i++)
            {
                halfArray[i] = (Half)floatArray[i];
            }
            return halfArray.Select(ToFloat16).ToArray();
#elif NET461 || NET472 || NET48 || NETSTANDARD2_0 || NETSTANDARD2_1 || NETCOREAPP3_1
            // 不支持原生Half类型的框架，直接转换为半精度表示
            Float16[] result = new Float16[floatArray.Length];
            for (int i = 0; i < floatArray.Length; i++)
            {
                result[i] = FloatToFloat16(floatArray[i]);
            }
            return result;
#endif
        }

#if NET461 || NET472 || NET48 || NETSTANDARD2_0 || NETSTANDARD2_1 || NETCOREAPP3_1
        /// <summary>
        /// float转换为Float16（适用于不支持原生Half的框架）
        /// </summary>
        private static Float16 FloatToFloat16(float value)
        {
            // 处理特殊值
            if (float.IsNaN(value))
                return (Float16)0x7FFF; // NaN的半精度表示
            if (float.IsPositiveInfinity(value))
                return (Float16)0x7C00; // 正无穷
            if (float.IsNegativeInfinity(value))
                return (Float16)0xFC00; // 负无穷
            if (value == 0.0f)
                return (Float16)(GetSignBit(value) ? 0x8000 : 0x0000);

            // 替代BitConverter.SingleToInt32Bits的实现
            int floatBits = GetFloatBits(value);
            
            int sign = (floatBits >> 31) & 0x1;
            int exponent = (floatBits >> 23) & 0xFF;
            int mantissa = floatBits & 0x7FFFFF;

            // 半精度参数计算
            int halfExponent = exponent - 127 + 15;
            int halfMantissa = mantissa >> 13;

            // 处理溢出情况
            if (halfExponent > 31)
                return (Float16)((sign << 15) | 0x7C00); // 无穷大
            if (halfExponent < 0)
                return (Float16)(sign << 15); // 下溢为0

            // 组合结果
            ushort halfBits = (ushort)((sign << 15) | (halfExponent << 10) | halfMantissa);
            return (Float16)halfBits;
        }

        /// <summary>
        /// 获取float的符号位（旧框架兼容版）
        /// </summary>
        private static bool GetSignBit(float value)
        {
            unsafe
            {
                return *(int*)&value < 0;
            }
        }

        /// <summary>
        /// 获取float的位表示（替代SingleToInt32Bits，旧框架兼容版）
        /// </summary>
        private static int GetFloatBits(float value)
        {
            unsafe
            {
                return *(int*)&value;
            }
        }
#endif

#if  NET5_0_OR_GREATER// ||NETSTANDARD2_1 || NETCOREAPP3_1
        /// <summary>
        /// 原生Half类型转换为Float16（仅在支持Half的框架下可用）
        /// </summary>
        public static Float16 ToFloat16(Half half)
        {
            FloatConverter2 converter = new FloatConverter2();
            converter.half = half;
            return (Float16)converter.ushortValue;
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct FloatConverter2
        {
            [FieldOffset(0)]
            public Half half;

            [FieldOffset(0)]
            public ushort ushortValue;
        }
#endif
    }
}
