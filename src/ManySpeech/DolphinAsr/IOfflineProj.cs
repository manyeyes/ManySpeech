using ManySpeech.DolphinAsr.Model;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.DolphinAsr
{
    internal interface IOfflineProj
    {
        InferenceSession EncoderSession
        {
            get;
            set;
        }
        InferenceSession DecoderSession
        {
            get;
            set;
        }
        OfflineModel OfflineModel
        {
            get;
            set;
        }        
        List<float[]> stack_states(List<List<float[]>> statesList);
        List<List<float[]>> unstack_states(List<float[]> states);
        internal EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs);
        internal DecoderOutputEntity DecoderProj(List<List<int>> tokensList, float[] encoder_outputs);
        internal List<List<int>> DecodeAsr(Tensor<float> logitsTensor);
        internal List<List<int>> DetectLanguage(Tensor<float> logitsTensor);
        internal List<List<int>> DetectRegion(Tensor<float> logitsTensor);
        internal void Dispose();
    }
}
