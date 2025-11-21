// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.WhisperAsr.Model
{
    public class DecodingResultEntity
    {
        private string? _language;
        private Dictionary<string,float>? _languageProbs;
        private List<int> _tokens=new List<int>();
        private string? _text;
        private float _avgLogprob = float.NaN;
        private float _noSpeechProb = float.NaN;
        private float _temperature = float.NaN;
        private float _compressionRatio=float.NaN;

        public string? Language { get => _language; set => _language = value; }
        public Dictionary<string, float>? LanguageProbs { get => _languageProbs; set => _languageProbs = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public string? Text { get => _text; set => _text = value; }
        public float AvgLogprob { get => _avgLogprob; set => _avgLogprob = value; }
        public float NoSpeechProb { get => _noSpeechProb; set => _noSpeechProb = value; }
        public float Temperature { get => _temperature; set => _temperature = value; }
        public float CompressionRatio { get => _compressionRatio; set => _compressionRatio = value; }
    }

    public class SegmentEntity
    {
        private int _seek;
        private float? _start;
        private float? _end;
        private string? _text;
        private string? _language;
        private List<int> _tokens=new List<int>();
        private float? _temperature;
        private float? _avgLogprob;
        private float? _compressionRatio;
        private float? _noSpeechProb;
        private List<Dictionary<string, string>>? _words;

        public int Seek { get => _seek; set => _seek = value; }
        public float? Start { get => _start; set => _start = value; }
        public float? End { get => _end; set => _end = value; }
        public string? Text { get => _text; set => _text = value; }
        public string? Language { get => _language; set => _language = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public float? Temperature { get => _temperature; set => _temperature = value; }
        public float? AvgLogprob { get => _avgLogprob; set => _avgLogprob = value; }
        public float? CompressionRatio { get => _compressionRatio; set => _compressionRatio = value; }
        public float? NoSpeechProb { get => _noSpeechProb; set => _noSpeechProb = value; }
        public List<Dictionary<string, string>>? Words { get => _words; set => _words = value; }
    }
}
