// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.WenetAsr.Model
{
    public class CtcOutputEntity
    {
        private float[]? _probs;
        private List<Int64[]>? hyps;
        private List<Int64>? hyps_lens;

        public float[]? Probs { get => _probs; set => _probs = value; }
        public List<Int64[]>? Hyps { get => hyps; set => hyps = value; }
        public List<Int64>? Hyps_lens { get => hyps_lens; set => hyps_lens = value; }
    }
}
