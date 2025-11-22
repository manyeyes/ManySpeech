using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.AliParaformerAsr
{
    internal interface IOfflineProj
    {
        InferenceSession ModelSession 
        {
            get;
            set;
        }
        int Blank_id
        {
            get;
            set;
        }
        int Sos_eos_id
        {
            get;
            set;
        }
        int Unk_id
        {
            get;
            set;
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
        internal ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs);
        internal void Dispose();
    }
}
