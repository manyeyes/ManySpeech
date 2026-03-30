// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr.Model
{
    public class EmbedOutputEntity
    {
        private Tensor<float>? _embedOut;

        public Tensor<float>? EmbedOut { get => _embedOut; set => _embedOut = value; }
    }
}
