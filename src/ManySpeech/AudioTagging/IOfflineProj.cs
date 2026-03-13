using Microsoft.ML.OnnxRuntime;
using ManySpeech.AudioTagging.Model;

namespace ManySpeech.AudioTagging
{
    internal interface IOfflineProj
    {
        InferenceSession ModelSession
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
        int Required_cache_size
        {
            get;
            set;
        }
        List<float[]> stack_states(List<List<float[]>> statesList);
        List<List<float[]>> unstack_states(List<float[]> states);
        internal ModelOutputEntity ModelProj(List<ModelInputEntity> modelInputs);
        internal void Dispose();
    }
}
