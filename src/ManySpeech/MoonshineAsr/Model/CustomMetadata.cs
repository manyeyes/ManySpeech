// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.MoonshineAsr.Model
{
    public class CustomMetadata
    {
        //model metadata
        private int _dim=416;
        private int _inner_dim=416;
        private int _n_head=8;
        private int _enc_n_layers=8;
        private int _dec_n_layers = 8;
        private string? _model_specs= "base";

        public int dim { get => _dim; set => _dim = value; }
        public int inner_dim { get => _inner_dim; set => _inner_dim = value; }
        public int n_head { get => _n_head; set => _n_head = value; }
        public int enc_n_layers { get => _enc_n_layers; set => _enc_n_layers = value; }
        public int dec_n_layers { get => _dec_n_layers; set => _dec_n_layers = value; }
        public string? model_specs { get => _model_specs; set => _model_specs = value; }
    }
}
