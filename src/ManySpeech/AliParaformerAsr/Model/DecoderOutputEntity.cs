// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace ManySpeech.AliParaformerAsr.Model
{
    public class DecoderOutputEntity
    {
        private float[]? _logits;
        private List<Int64[]>? _sample_ids;
        private List<float[]> statesList;

        public float[]? Logits { get => _logits; set => _logits = value; }
        public List<long[]>? Sample_ids { get => _sample_ids; set => _sample_ids = value; }
        public List<float[]> StatesList { get => statesList; set => statesList = value; }
    }
}
