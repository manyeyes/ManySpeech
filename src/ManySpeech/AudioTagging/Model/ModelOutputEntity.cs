// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AudioTagging.Model
{
    public class ModelOutputEntity
    {
        private Tensor<float>? _modelOut;

        public Tensor<float>? ModelOut { get => _modelOut; set => _modelOut = value; }
    }
}
