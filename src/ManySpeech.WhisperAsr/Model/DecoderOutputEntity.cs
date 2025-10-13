// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.WhisperAsr.Model
{
    public class DecoderOutputEntity
    {
        private Tensor<float>? _logits;
        private int[]? _dim;

        public Tensor<float>? Logits { get => _logits; set => _logits = value; }
        public int[]? Dim { get => _dim; set => _dim = value; }
    }
}
