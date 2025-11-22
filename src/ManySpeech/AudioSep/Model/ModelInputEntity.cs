// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.AudioSep.Model
{
    public class ModelInputEntity
    {
        private string? _audioId=string.Empty;
        private float[]? _speech;
        private int _speechLength=0;
        private int _sampleRate = 16000;
        private int _channels = 2;

        public string? AudioId { get => _audioId; set => _audioId = value; }
        public float[]? Speech { get => _speech; set => _speech = value; }
        public int SpeechLength { get => _speechLength; set => _speechLength = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Channels { get => _channels; set => _channels = value; }
    }
}
