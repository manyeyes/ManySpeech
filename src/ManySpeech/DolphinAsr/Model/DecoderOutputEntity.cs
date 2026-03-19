// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.DolphinAsr.Model
{
    public class DecoderOutputEntity
    {
        private Tensor<float>? _logitsTensor;
        private float[]? _logits;
        private List<List<int>> _tokenIdsList = new List<List<int>>();
        private List<float> _rescoring_score = new List<float>();

        public float[]? Logits { get => _logits; set => _logits = value; }
        public List<float> Rescoring_score { get => _rescoring_score; set => _rescoring_score = value; }
        public List<List<int>> TokenIdsList { get => _tokenIdsList; set => _tokenIdsList = value; }
        public Tensor<float>? LogitsTensor { get => _logitsTensor; set => _logitsTensor = value; }
    }
}
