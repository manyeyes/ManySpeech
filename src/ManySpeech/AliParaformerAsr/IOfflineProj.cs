using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.SeqUnit;

namespace ManySpeech.AliParaformerAsr
{
    internal interface IOfflineProj
    {
        OfflineModel OfflineModel 
        { 
            get; 
            //set; 
        }
        ITokenizer Tokenizer 
        { 
            get; 
            //set; 
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
        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<string>? languages = null, List<string>? regions = null);
        internal void Dispose();
    }
}
