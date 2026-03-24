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

namespace ManySpeech.AliParaformerAsr.Utils
{
    internal static class ComputeHelper
    {
        public static List<int[]> TimestampLfr6(float[] us_cif_peak, int[] tokens, float begin_time = 0.0F, float total_offset = -1.5F)
        {
            List<float[]> timestamp_list = new List<float[]>();
            int START_END_THRESHOLD = 5;
            int MAX_TOKEN_DURATION = 30;
            float TIME_RATE = 10.0F * 6 / 1000 / 3;//3 times upsampled

            int num_frames = us_cif_peak.Length;
            if (tokens.Last() == 2)
            {
                int[] newTokens = new int[tokens.Length - 1];
                Array.Copy(tokens, 0, newTokens, 0, newTokens.Length);
                tokens = newTokens;
            }
            List<float> fire_place = new List<float>();
            for (int i = 0; i < us_cif_peak.Length; i++)
            {
                if (us_cif_peak[i] > 1.0F - 1e-4)
                {
                    fire_place.Add(i + total_offset);
                }
            }
            List<bool> new_char_list = new List<bool>();
            //begin silence
            if (fire_place[0] > START_END_THRESHOLD)
            {
                timestamp_list.Add(new float[] { 0.0f, fire_place[0] * TIME_RATE });
                new_char_list.Add(false);
            }
            // tokens timestamp
            List<int[]> timestamps = new List<int[]>();
            for (int i = 0; i < fire_place.Count - 1; i++)
            {
                if (tokens[i] == 1)
                {
                    new_char_list.Add(false);
                }
                else
                {
                    new_char_list.Add(true);
                }
                if (i == fire_place.Count - 2 || MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION)
                {
                    timestamp_list.Add(new float[] { fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE });
                }
                else
                {
                    float _split = fire_place[i] + MAX_TOKEN_DURATION;
                    timestamp_list.Add(new float[] { fire_place[i] * TIME_RATE, _split * TIME_RATE });
                    timestamp_list.Add(new float[] { _split * TIME_RATE, fire_place[i + 1] * TIME_RATE });
                    new_char_list.Add(false);
                }
            }
            // tail token and end silence
            if (num_frames - fire_place.Last() > START_END_THRESHOLD)
            {
                float _end = (float)((num_frames + fire_place.Last()) / 2);
                timestamp_list.Last()[1] = _end * TIME_RATE;
                timestamp_list.Add(new float[] { _end * TIME_RATE, num_frames * TIME_RATE });
                new_char_list.Add(false);
            }
            else
            {
                timestamp_list.Last()[1] = num_frames * TIME_RATE;
            }
            if (begin_time > 0.0F)
            {
                for (int i = 0; i < timestamp_list.Count; i++)
                {
                    timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0F;
                    timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0F;
                }
            }
            new_char_list.Add(true);
#if NET6_0_OR_GREATER
            // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
            foreach (var item in new_char_list.Zip<bool, float[]>(timestamp_list))
            {
                bool charX = item.First;
                float[] timestamp = item.Second;
                if (charX)
                {
                    timestamps.Add(new int[] { (int)(timestamp[0] * 1000), (int)(timestamp[1] * 1000) });
                }
            }
#else
            // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
            for (int i = 0; i < new_char_list.Count && i < timestamp_list.Count; i++)
            {
                bool charX = new_char_list[i];
                float[] timestamp = timestamp_list[i];

                if (charX)
                {
                    timestamps.Add(new int[] {
                        (int)(timestamp[0] * 1000),
                        (int)(timestamp[1] * 1000)
                    });
                }
            }
#endif
            return timestamps;
        }
    }
}
