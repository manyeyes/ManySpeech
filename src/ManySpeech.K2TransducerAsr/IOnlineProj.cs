// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.K2TransducerAsr
{
    public interface IOnlineProj
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
        InferenceSession JoinerSession
        {
            get;
            set;
        }
        OnlineCustomMetadata CustomMetadata
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
        List<List<float[]>> GetEncoderInitStates(int batchSize = 1);
        List<List<float[]>> stack_states(List<List<List<float[]>>> stateList);
        List<List<List<float[]>>> unstack_states(List<float[]> encoder_out_states);
        internal EncoderOutputEntity EncoderProj(List<OnlineInputEntity> modelInputs, int batchSize, List<List<float[]>> statesList);
        internal DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize);
        internal JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out);
        internal void Dispose();
    }
}