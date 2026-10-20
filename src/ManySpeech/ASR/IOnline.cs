// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.ASR.Model;
using ManySpeech.SeqUnit;

namespace ManySpeech.ASR
{
    public interface IOnline
    {
        OnlineModel OnlineModel
        {
            get;
        }
        ITokenizer Tokenizer
        {
            get;
        }
        int SampleRate
        {
            get;
            set;
        }
        int FeatureDim
        {
            get;
            set;
        }
        int SpeechLength
        {
            get;
            set;
        }

        bool IsResizeAudioDuration
        {
            get;
            set;
        }
        bool IsPaddingSpeech
        {
            get;
            set;
        }
        bool IsSampleScalingRequired
        {
            get;
            set;
        }
        int ChunkLength
        {
            get;
            set;
        }
        int ShiftLength
        {
            get;
            set;
        }
        public void Infer(List<OnlineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<int> numTrailingBlanks, List<int> frameOffsets, List<string>? languages = null, List<string>? regions = null, List<List<List<float[]>>>? statesList = null);
        public List<int> GetDecoderInitTokenIds();
        public float[] GetDecoderInitCacheFeats();
        public List<List<float[]>>? GetDecoderInitCaches();
        internal void Dispose();
    }
}