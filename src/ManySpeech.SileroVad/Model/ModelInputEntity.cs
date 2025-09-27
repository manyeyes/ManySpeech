// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.SileroVad.Model
{
    public class ModelInputEntity
    {
        private float[]? _speech;
        private int _speech_length;
        public float[]? Speech { get => _speech; set => _speech = value; }
        public int SpeechLength { get => _speech_length; set => _speech_length = value; }
    }
}
