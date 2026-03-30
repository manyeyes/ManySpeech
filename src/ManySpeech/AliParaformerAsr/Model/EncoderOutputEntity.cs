// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr.Model
{
    public class EncoderOutputEntity
    {
        private Tensor<float>? _encOut;
        private Int64[]? _encOutLens;

        private List<List<float[]>>? _enc;
        private int[]? _enc_len;
        private List<List<float>>? _alphas;

        public List<List<float[]>>? Enc { get => _enc; set => _enc = value; }
        public int[]? Enc_len { get => _enc_len; set => _enc_len = value; }
        public List<List<float>>? Alphas { get => _alphas; set => _alphas = value; }
        public Tensor<float>? EncOut { get => _encOut; set => _encOut = value; }
        public Int64[]? EncOutLens { get => _encOutLens; set => _encOutLens = value; }
    }
}
