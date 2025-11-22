using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.AliParaformerAsr.Model
{
    internal class ModelOutputEntity
    {
        private Tensor<float>? _model_out;
        private int[]? _model_out_lens;
        private Tensor<float>? _cif_peak_tensor;

        public Tensor<float>? model_out { get => _model_out; set => _model_out = value; }
        public int[]? model_out_lens { get => _model_out_lens; set => _model_out_lens = value; }
        public Tensor<float>? cif_peak_tensor { get => _cif_peak_tensor; set => _cif_peak_tensor = value; }
    }
}
