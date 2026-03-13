using ManySpeech.OmniAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.OmniAsr
{
    public interface IOfflineProj
    {
        InferenceSession ModelSession
        {
            get;
            set;
        }
        int FeatureDim
        {
            get;
            set;
        }
        int SampleRate
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
        internal OfflineOutputEntity ModelProj(List<OfflineInputEntity> modelInputs);

        internal void Dispose();
    }
}
