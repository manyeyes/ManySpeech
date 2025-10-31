using Microsoft.ML.OnnxRuntime;
using ManySpeech.AudioSep.Model;

namespace ManySpeech.AudioSep
{
    internal interface ISepProj
    {
        InferenceSession ModelSession
        {
            get;
            set;
        }
        CustomMetadata CustomMetadata
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
        int Channels
        {
            get;
            set;
        }
        List<ModelOutputEntity> ModelProj(List<ModelInputEntity> modelInputs, List<float[]>? statesList=null, int offset=0);
        List<ModelOutputEntity> GeneratorProj(ModelOutputEntity modelOutputEntity, int batchSize = 1);
        void Dispose();
    }
}
