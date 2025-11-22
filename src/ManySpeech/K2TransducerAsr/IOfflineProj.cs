using ManySpeech.K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.K2TransducerAsr
{
    public interface IOfflineProj
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
        OfflineCustomMetadata CustomMetadata
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
        internal EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs, int batchSize);
        internal DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize);
        internal JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out);

        internal void Dispose();
    }
}
