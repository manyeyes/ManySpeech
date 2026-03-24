// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.AliParaformerAsr.Model
{
    /// <summary>
    /// Represents the result of an speech recognition operation.
    /// </summary>
    public class OfflineRecognizerResultEntity
    {
        private string? _region;
        private string? _language;
        private List<string>? _tokens = new List<string>();
        private List<int[]>? _timestamps = new List<int[]>();
        private string? _text;
        private List<string>? _words = new List<string>();
        private List<int[]>? _wordsTimestamps = new List<int[]>();
        private List<SegmentEntity> _segments = new List<SegmentEntity>();


        /// <summary>
        /// Gets or sets the list of decoded token strings.
        /// </summary>
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }


        /// <summary>
        /// Gets or sets the timestamps for each token (start and end times).
        /// </summary>
        public List<int[]>? Timestamps { get => _timestamps; set => _timestamps = value; }


        /// <summary>
        /// Gets or sets the detected language.
        /// </summary>
        public string? Language { get => _language; set => _language = value; }


        /// <summary>
        /// Gets or sets the full transcribed text.
        /// </summary>
        public string? Text { get => _text; set => _text = value; }


        /// <summary>
        /// Gets or sets the detected region or dialect.
        /// </summary>
        public string? Region { get => _region; set => _region = value; }


        /// <summary>
        /// Gets or sets the list of word-level strings.
        /// </summary>
        public List<string>? Words { get => _words; set => _words = value; }


        /// <summary>
        /// Gets or sets the timestamps for each word (start and end times).
        /// </summary>
        public List<int[]>? WordsTimestamps { get => _wordsTimestamps; set => _wordsTimestamps = value; }


        /// <summary>
        /// Gets or sets the list of segment entities, each containing detailed information for a speech segment.
        /// </summary>
        public List<SegmentEntity> Segments { get => _segments; set => _segments = value; }
    }


    /// <summary>
    /// Represents a segment of transcribed speech with detailed metadata.
    /// </summary>
    public class SegmentEntity
    {
        private int _seek;
        private float? _start;
        private float? _end;
        private string? _text;
        private string? _language;
        private List<string>? _tokens = new List<string>();
        private List<int> _tokenIds = new List<int>();
        private List<Dictionary<string, string>>? _words;
        private float? _temperature;
        private float? _avgLogprob;
        private float? _compressionRatio;
        private float? _noSpeechProb;


        /// <summary>
        /// Gets or sets the seek offset (in frames) for this segment.
        /// </summary>
        public int Seek { get => _seek; set => _seek = value; }


        /// <summary>
        /// Gets or sets the start time of the segment (in seconds).
        /// </summary>
        public float? Start { get => _start; set => _start = value; }


        /// <summary>
        /// Gets or sets the end time of the segment (in seconds).
        /// </summary>
        public float? End { get => _end; set => _end = value; }


        /// <summary>
        /// Gets or sets the transcribed text of the segment.
        /// </summary>
        public string? Text { get => _text; set => _text = value; }


        /// <summary>
        /// Gets or sets the language of the segment.
        /// </summary>
        public string? Language { get => _language; set => _language = value; }


        /// <summary>
        /// Gets or sets the list of token strings for this segment.
        /// </summary>
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }


        /// <summary>
        /// Gets or sets the list of token IDs for this segment.
        /// </summary>
        public List<int> TokenIds { get => _tokenIds; set => _tokenIds = value; }


        /// <summary>
        /// Gets or sets the word-level details, each as a dictionary of properties (e.g., "word", "start", "end").
        /// </summary>
        public List<Dictionary<string, string>>? Words { get => _words; set => _words = value; }


        /// <summary>
        /// Gets or sets the temperature used during decoding for this segment.
        /// </summary>
        public float? Temperature { get => _temperature; set => _temperature = value; }


        /// <summary>
        /// Gets or sets the average log probability of the segment's tokens.
        /// </summary>
        public float? AvgLogprob { get => _avgLogprob; set => _avgLogprob = value; }


        /// <summary>
        /// Gets or sets the compression ratio of the segment (text length / token length).
        /// </summary>
        public float? CompressionRatio { get => _compressionRatio; set => _compressionRatio = value; }


        /// <summary>
        /// Gets or sets the probability of no speech being present in this segment.
        /// </summary>
        public float? NoSpeechProb { get => _noSpeechProb; set => _noSpeechProb = value; }
    }
}
