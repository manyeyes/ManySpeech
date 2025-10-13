using ManySpeech.WhisperAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.WhisperAsr
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
        CustomMetadata CustomMetadata
        {
            get;
            set;
        }
        //ConfEntity? ConfEntity
        //{
        //    get;
        //    set;
        //}
        int SampleRate
        {
            get;
            set;
        }
        int FeatureDim
        {
            get;
            set;
        }
        int ChunkLength
        {
            get;
            set;
        }
        int FrameLength
        {
            get;
            set;
        }
        int ShiftLength
        {
            get;
            set;
        }
        int HopLength
        {
            get;
            set;
        }
        internal EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs);
        internal DecoderOutputEntity DetectLanguage(EncoderOutputEntity encoderOutputEntity, Int64 tokenizerSot);
        internal DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, List<List<Int64>> tokens);
        internal void Dispose();
    }
}
