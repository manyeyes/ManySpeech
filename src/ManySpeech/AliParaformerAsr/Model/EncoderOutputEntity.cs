// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Collections;

namespace ManySpeech.AliParaformerAsr.Model
{
    public class EncoderOutputEntity
    {

        private List<List<float[]>>? _enc;
        private int[]? _enc_len;
        private List<List<float>>? _alphas;

        public List<List<float[]>>? Enc { get => _enc; set => _enc = value; }
        public int[]? Enc_len { get => _enc_len; set => _enc_len = value; }
        public List<List<float>>? Alphas { get => _alphas; set => _alphas = value; }
    }
}
