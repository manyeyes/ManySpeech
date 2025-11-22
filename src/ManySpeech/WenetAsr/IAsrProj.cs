using Microsoft.ML.OnnxRuntime;
using ManySpeech.WenetAsr.Model;

namespace ManySpeech.WenetAsr
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
        InferenceSession CtcSession
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
        internal EncoderOutputEntity EncoderProj(List<AsrInputEntity> modelInputs, List<float[]> statesList, int offset);
        internal DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, CtcOutputEntity ctcOutputEntity, int batchSize = 1);
        internal CtcOutputEntity CtcProj(EncoderOutputEntity encoderOutput);
        internal void Dispose();
    }
}
