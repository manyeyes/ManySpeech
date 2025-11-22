// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.WhisperAsr.Model
{
    public class OfflineCustomMetadata
    {
        //encoder metadata
        private string _version;
        private string _model_type;
        private string _model_author;
        private int _context_size = 2;
        private int _vocab_size = 500;
        private string? _comment;
        public string Version { get => _version; set => _version = value; }
        public string Model_type { get => _model_type; set => _model_type = value; }
        public string Model_author { get => _model_author; set => _model_author = value; }
        public int Context_size { get => _context_size; set => _context_size = value; }
        public int Vocab_size { get => _vocab_size; set => _vocab_size = value; }
        public string? Comment { get => _comment; set => _comment = value; }
    }
}
