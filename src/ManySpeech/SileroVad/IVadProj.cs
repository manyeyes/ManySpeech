using Microsoft.ML.OnnxRuntime;
using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad
{
    internal interface IVadProj
    {
        InferenceSession ModelSession
        {
            get;
            set;
        }
        ModelCustomMetadata CustomMetadata 
        { 
            get; 
            set; 
        }
        int SampleRate
        {
            get;
            set;
        }
        bool IsDebug
        {
            get;
            set;
        }

        internal List<float[]> GetModelInitStates(int batchSize = 1);
        internal List<float[]> stack_states_unittest(List<List<float[]>> stateList);
        internal List<List<float[]>> unstack_states_unittest(List<float[]> stateList);
        internal List<float[]> stack_states(List<List<float[]>> stateList);
        internal List<List<float[]>> unstack_states(List<float[]> modelOutStates);
        internal ModelOutputEntity ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList);
        internal void Dispose();
    }
}
