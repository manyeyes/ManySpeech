// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes

namespace ManySpeech.FireRedAsr.Model
{
    public class DecoderOutputEntity
    {
        private float[]? _logits;
        private List<List<Int64>> _tokensList;
        private List<float> _rescoring_score;
        private List<float[]> _cacheList;

        public float[]? Logits { get => _logits; set => _logits = value; }
        public List<float[]> CacheList { get => _cacheList; set => _cacheList = value; }
        public List<float> Rescoring_score { get => _rescoring_score; set => _rescoring_score = value; }
        public List<List<long>> TokensList { get => _tokensList; set => _tokensList = value; }
    }
}
