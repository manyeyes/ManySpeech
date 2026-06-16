// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.ASR.Model
{
    public class DecoderOutputEntity
    {
        private Tensor<float>? _logitsTensor;
        private Int64[]? _decOutLens;

        private float[]? _logits;
        private List<Int64[]>? _sample_ids;
        private List<float[]> statesList;

        public float[]? Logits { get => _logits; set => _logits = value; }
        public List<long[]>? Sample_ids { get => _sample_ids; set => _sample_ids = value; }
        public List<float[]> StatesList { get => statesList; set => statesList = value; }
        public Tensor<float>? LogitsTensor { get => _logitsTensor; set => _logitsTensor = value; }
        public Int64[]? DecOutLens { get => _decOutLens; set => _decOutLens = value; }
    }
}
