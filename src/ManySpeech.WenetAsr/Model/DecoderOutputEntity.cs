// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace ManySpeech.WenetAsr.Model
{
    public class DecoderOutputEntity
    {
        private float[]? _logits;
        private List<Int64[]>? _sample_ids;
        private List<Int64[]>? _hyps;
        private List<Int64>? _hyps_lens;
        private List<float> _rescoring_score;

        public float[]? Logits { get => _logits; set => _logits = value; }
        public List<long[]>? Sample_ids { get => _sample_ids; set => _sample_ids = value; }
        public List<long[]>? Hyps { get => _hyps; set => _hyps = value; }
        public List<long>? Hyps_lens { get => _hyps_lens; set => _hyps_lens = value; }
        public List<float> Rescoring_score { get => _rescoring_score; set => _rescoring_score = value; }
    }
}
