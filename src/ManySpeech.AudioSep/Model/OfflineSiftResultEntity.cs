// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.AudioSep.Model
{
    /// <summary>
    /// online recognizer result entity 
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class OfflineSepResultEntity
    {
        private string _audioId=string.Empty;
        private Dictionary<string, float[]?> _stems=new Dictionary<string, float[]?>();
        private int _channels = 2;
        private int _sampleRate = 16000;
        private List<int[]>? _timestamps = new List<int[]>();

        public string AudioId { get => _audioId; set => _audioId = value; }
        public Dictionary<string, float[]?> Stems { get => _stems; set => _stems = value; }
        public int Channels { get => _channels; set => _channels = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public List<int[]>? Timestamps { get => _timestamps; set => _timestamps = value; }
    }
}
