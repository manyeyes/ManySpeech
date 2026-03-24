using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr.Model
{
    internal class ModelOutputEntity
    {
        private Tensor<float>? _modelOut;
        private int[]? _modelOutLens;
        private Tensor<float>? _cifPeak;

        public Tensor<float>? ModelOut { get => _modelOut; set => _modelOut = value; }
        public int[]? ModelOutLens { get => _modelOutLens; set => _modelOutLens = value; }
        public Tensor<float>? CifPeak { get => _cifPeak; set => _cifPeak = value; }
    }
}
