// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.FireRedAsr.Model
{
    public class EncoderOutputEntity
    {
        private Tensor<float>? _encOut;
        private Int64[]? _encOutLens;
        private bool[]? _mask;
        public List<float[]> CrossKVList { get; set; }

        public Tensor<float>? EncOut { get => _encOut; set => _encOut = value; }
        public Int64[]? EncOutLens { get => _encOutLens; set => _encOutLens = value; }
        public bool[]? Mask { get => _mask; set => _mask = value; }
    }
}
