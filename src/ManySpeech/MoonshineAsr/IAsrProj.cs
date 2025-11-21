using Microsoft.ML.OnnxRuntime;
using ManySpeech.MoonshineAsr.Model;

namespace ManySpeech.MoonshineAsr
{
    internal interface IAsrProj
    {
        InferenceSession PreprocessSession
        {
            get;
            set;
        }
        InferenceSession EncodeSession
        {
            get;
            set;
        }
        InferenceSession CachedDecodeSession
        {
            get;
            set;
        }
        InferenceSession UncachedDecodeSession
        {
            get;
            set;
        }
        CustomMetadata CustomMetadata
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
        internal PreprocessOutputEntity PreprocessProj(List<AsrInputEntity> modelInputs);
        internal EncodeOutputEntity EncodeProj(PreprocessOutputEntity preprocessOutputEntity);
        internal UncachedDecodeOutputEntity UncachedDecodeProj(EncodeOutputEntity encodeOutputEntity, List<int[]> nextTokens, List<int> seqLens);
        internal CachedDecodeOutputEntity CachedDecodeProj(EncodeOutputEntity encodeOutputEntity, List<int[]> nextTokens, List<int> seqLens, List<float[]> cacheList);
        void Dispose();
    }
}
