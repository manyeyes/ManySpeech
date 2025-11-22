// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace ManySpeech.K2TransducerAsr.Model
{
    /// <summary>
    /// offline custom metadata entity 
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineCustomMetadata
    {
        private string? _version;
        private string? _model_type;
        private string? _model_author;
        private int _context_size = 2;
        private int _vocab_size = 500;
        private int _joiner_dim;
        private string? _comment;
        private string _feature_type="fbank";

        public string? Version { get => _version; set => _version = value; }
        public string? Model_type { get => _model_type; set => _model_type = value; }
        public string? Model_author { get => _model_author; set => _model_author = value; }
        public int Context_size { get => _context_size; set => _context_size = value; }
        public int Vocab_size { get => _vocab_size; set => _vocab_size = value; }
        public int Joiner_dim { get => _joiner_dim; set => _joiner_dim = value; }
        public string? Comment { get => _comment; set => _comment = value; }
        public string Feature_type { get => _feature_type; set => _feature_type = value; }
    }
}
