using ManySpeech.AudioSep.Model;

namespace ManySpeech.AudioSep.Utils
{
    internal static class PadHelper
    {
        public static float[] PadSequence(List<ModelInputEntity> modelInputs)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats);
        }
        public static float[] PadSequence(List<ModelInputEntity> modelInputs, int tailLen = 0)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats, tailLen: tailLen);
        }

        private static float[] PadSequence(List<float[]?> floats, int tailLen = 0)
        {
            int max_speech_length = floats.Where(x => x != null).Max(x => x.Length) + tailLen;
            int speech_length = max_speech_length * floats.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[floats.Count, max_speech_length];
            for (int i = 0; i < floats.Count; i++)
            {
                if (floats[i] == null || max_speech_length == floats[i].Length)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                        xxx[i, j] = floats[i][j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - floats[i].Length];
                float[]? curr_speech = floats[i];
                float[] padspeech = new float[max_speech_length];
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                //Array.Copy(nullspeech, 0, padspeech, curr_speech.Length, nullspeech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    xxx[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。 
                }
            }
            //Array.Copy(xxx, 0, speech, 0, speech.Length);//one len is 3120
            int s = 0;
            for (int i = 0; i < xxx.GetLength(0); i++)
            {
                for (int j = 0; j < xxx.GetLength(1); j++)
                {
                    speech[s] = xxx[i, j];
                    s++;
                }
            }
            return speech;
        }

        public static float[] PadSequence_unittest(List<ModelInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? curr_speech = modelInputs[i].Speech;
                Array.Copy(curr_speech, 0, speech, i * curr_speech.Length, curr_speech.Length);
            }
            return speech;
        }
    }
}
