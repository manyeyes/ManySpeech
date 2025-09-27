using Microsoft.ML.OnnxRuntime;
using ManySpeech.FireRedAsr.Model;

namespace ManySpeech.FireRedAsr
{
    internal interface IAsrProj
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
        int Unk_id
        {
            get;
            set;
        }
        int Sos_id
        {
            get;
            set;
        }
        int Eos_id
        {
            get;
            set;
        }
        int Pad_id
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
        internal EncoderOutputEntity EncoderProj(List<AsrInputEntity> modelInputs);
        internal DecoderOutputEntity DecoderProj(List<List<Int64>> tokensList, float[] encoder_outputs, bool[] src_mask, List<float[]> cacheList);
        internal void Dispose();
    }
}
