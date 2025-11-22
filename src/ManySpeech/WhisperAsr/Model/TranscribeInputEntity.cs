// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.WhisperAsr.Model
{
    public class TranscribeInputEntity
    {
        public float[]? Speech { get; set; }
        public int SpeechLength { get; set; }
        public int SampleLength { get; set; }
    }
}
