// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.ASR.Model;
using ManySpeech.SeqUnit;

namespace ManySpeech.ASR
{
    internal interface IOffline
    {
        OfflineModel OfflineModel
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
        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<string>? languages = null, List<string>? regions = null, List<List<float[]>>? statesList = null);
        public List<int> GetDecoderInitTokenIds();
        public List<float[]>? GetDecoderInitCaches();
        internal void Dispose();
    }
}
