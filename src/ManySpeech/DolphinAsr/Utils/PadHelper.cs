using ManySpeech.DolphinAsr.Model;

namespace ManySpeech.DolphinAsr.Utils
{
    internal static class PadHelper
    {
        public static float[] PadSequence(List<OfflineInputEntity> modelInputs)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats);
        }
        public static float[] PadSequence(List<OfflineInputEntity> modelInputs, int tailLen = 0)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats, tailLen: tailLen);
        }

        private static float[] PadSequence(List<float[]> floats, int tailLen = 0)
        {
            int max_speech_length = floats.Where(x => x != null).Max(x => x.Length) + 80 * tailLen;
            int speech_length = max_speech_length * floats.Count;
            float[] speech = new float[speech_length];
            // 填充极小值（可选择交替符号避免直流偏移）
            float epsilon = 1e-3f;
            for (int i = 0; i < speech.Length; i++)
            {
                // 每隔一个采样点取反，减少直流分量
                speech[i] = (i % 2 == 0) ? epsilon : -epsilon;
            }
            float[,] intermediate = new float[floats.Count, max_speech_length];
            for (int i = 0; i < floats.Count; i++)
            {
                if (floats[i] == null || max_speech_length == floats[i].Length)
                {
                    for (int j = 0; j < intermediate.GetLength(1); j++)
                    {
                        intermediate[i, j] = floats[i][j];
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - floats[i].Length];
                float[]? curr_speech = floats[i];
                float[] padspeech = new float[max_speech_length];
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    intermediate[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。 
                }
            }
            int s = 0;
            for (int i = 0; i < intermediate.GetLength(0); i++)
            {
                for (int j = 0; j < intermediate.GetLength(1); j++)
                {
                    speech[s] = intermediate[i, j];
                    s++;
                }
            }
            return speech;
        }

        public static float[] PadSequence_unittest(List<OfflineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            // 填充极小值（可选择交替符号避免直流偏移）
            float epsilon = 1e-3f;
            for (int i = 0; i < speech.Length; i++)
            {
                // 每隔一个采样点取反，减少直流分量
                speech[i] = (i % 2 == 0) ? epsilon : -epsilon;
            }
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? curr_speech = modelInputs[i].Speech;
                Array.Copy(curr_speech, 0, speech, i * curr_speech.Length, curr_speech.Length);
            }
            return speech;
        }
    }
}
