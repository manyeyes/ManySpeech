// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.WenetAsr.Model
{
    public class CustomMetadata
    {
        //model metadata
        private string _output_dir="";
        private int _output_size=512;
        private string? _onnx_infer;
        private int _left_chunks=-1;
        private int _batch=1;
        private float _reverse_weight=0.5f;
        private int _chunk_size;
        private int _num_blocks;
        private int _cnn_module_kernel;
        private int _head=8;
        private string? _decoder= "bitransformer";
        private int _feature_size;
        private int _vocab_size;
        private int _decoding_window;
        private string? _encoder="conformer";
        private int _subsampling_rate=4;
        private int _right_context;
        private bool _is_bidirectional_decoder=true;
        private int _eos_symbol;
        private int _sos_symbol;

        public string? Output_dir { get => _output_dir; set => _output_dir = value; }
        public int Output_size { get => _output_size; set => _output_size = value; }
        public string? Onnx_infer { get => _onnx_infer; set => _onnx_infer = value; }
        public int Left_chunks { get => _left_chunks; set => _left_chunks = value; }
        public int Batch { get => _batch; set => _batch = value; }
        public float Reverse_weight { get => _reverse_weight; set => _reverse_weight = value; }
        public int Chunk_size { get => _chunk_size; set => _chunk_size = value; }
        public int Num_blocks { get => _num_blocks; set => _num_blocks = value; }
        public int Cnn_module_kernel { get => _cnn_module_kernel; set => _cnn_module_kernel = value; }
        public int Head { get => _head; set => _head = value; }
        public string? Decoder { get => _decoder; set => _decoder = value; }
        public int Feature_size { get => _feature_size; set => _feature_size = value; }
        public int Vocab_size { get => _vocab_size; set => _vocab_size = value; }
        public int Decoding_window { get => _decoding_window; set => _decoding_window = value; }
        public string? Encoder { get => _encoder; set => _encoder = value; }
        public int Subsampling_rate { get => _subsampling_rate; set => _subsampling_rate = value; }
        public int Right_context { get => _right_context; set => _right_context = value; }
        public bool Is_bidirectional_decoder { get => _is_bidirectional_decoder; set => _is_bidirectional_decoder = value; }
        public int Eos_symbol { get => _eos_symbol; set => _eos_symbol = value; }
        public int Sos_symbol { get => _sos_symbol; set => _sos_symbol = value; }
    }
}
