using ManySpeech.WhisperAsr.Model;

namespace ManySpeech.WhisperAsr.Utils
{
    internal static class PadHelper
    {
        public static float[] PadSequence(List<OnlineInputEntity> modelInputs)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats);
        }
        public static float[] PadSequence(List<OfflineInputEntity> modelInputs, int tailLen = 19)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats, tailLen: tailLen);
        }
        public static float[] PadSequence(List<TranscribeInputEntity> modelInputs, int tailLen = 19)
        {
            List<float[]?> floats = modelInputs.Where(x => x != null).Select(x => x.Speech).ToList();
            return PadSequence(floats);
        }

        private static float[] PadSequence(List<float[]?> floats, int tailLen = 0)
        {
            int max_speech_length = floats.Where(x => x != null).Max(x => x.Length) + 80 * tailLen;
            int speech_length = max_speech_length * floats.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[floats.Count, max_speech_length];
            for (int i = 0; i < floats.Count; i++)
            {
                if (floats[i] == null || max_speech_length == floats[i].Length)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
                        xxx[i, j] = floats[i][j];
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
            speech = speech.Select(x => x == 0 ? -0.00070269317413142F : x).ToArray();
            return speech;
        }

        public static float[] PadSequence_unittest(List<OnlineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? curr_speech = modelInputs[i].Speech;
                Array.Copy(curr_speech, 0, speech, i * curr_speech.Length, curr_speech.Length);
            }
            speech = speech.Select(x => x == 0 ? -0.00070269317413142F : x).ToArray();
            return speech;
        }
    }
}
