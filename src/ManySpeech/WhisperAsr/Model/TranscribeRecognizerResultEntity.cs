// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.WhisperAsr.Model
{
    /// <summary>
    /// online recognizer result entity 
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class TranscribeRecognizerResultEntity
    {
        private string? _text = string.Empty;
        private List<int> _tokens=new List<int>();
        private List<SegmentEntity> _segments = new List<SegmentEntity>();
        //private List<int[]> _timestamps=new List<int[]>();
        private string? _language = string.Empty;

        public string? Text { get => _text; set => _text = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public List<SegmentEntity> Segments { get => _segments; set => _segments = value; }
        //public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public string? Language { get => _language; set => _language = value; }
    }
}
