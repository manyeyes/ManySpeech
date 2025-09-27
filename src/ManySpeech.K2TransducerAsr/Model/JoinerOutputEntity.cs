// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.K2TransducerAsr.Model
{
    public class JoinerOutputEntity
    {
        private float[]? logit;
        public float[]? Logit { get => logit; set => logit = value; }

        
        private Tensor<float> logits;
        public Tensor<float> Logits { get => logits; set => logits = value; }
    }
}
