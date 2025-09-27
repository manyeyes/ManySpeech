// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.K2TransducerAsr.Model
{
    /// <summary>
    /// online custom metadata entity 
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OnlineCustomMetadata
    {
        //encoder metadata
        private int[] _encoder_dims;
        private string? _version;
        private string? _model_type;
        private string? _model_author;
        private int[] _attention_dims;
        private int _decode_chunk_len = 0;
        private int[] _num_encoder_layers;
        private int _TT = 0;
        private int[] _cnn_module_kernels;
        private int[] _left_context_len;
        //decoder custom metadata
        private int _context_size = 2;
        private int _vocab_size = 500;
        //joiner custom metadata
        private int _joiner_dim;
        //zipformer2
        private int[] _query_head_dims;
        private string? _onnx_infer;
        private string? _comment;
        private int[] _value_head_dims;
        private int[] _num_heads;
        private string feature_type="fbank";
        //lstm
        private int _d_model;
        private int _rnn_hidden_size;
        //conformer
        private int _cnn_module_kernel;
        private int _pad_length;
        private int _encoder_dim;
        private int _chunk_size;
        private int _left_context;
        private int _right_context;

        public int[] Encoder_dims { get => _encoder_dims; set => _encoder_dims = value; }
        public string? Version { get => _version; set => _version = value; }
        public string? Model_type { get => _model_type; set => _model_type = value; }
        public string? Model_author { get => _model_author; set => _model_author = value; }
        public int[] Attention_dims { get => _attention_dims; set => _attention_dims = value; }
        public int Decode_chunk_len { get => _decode_chunk_len; set => _decode_chunk_len = value; }
        public int[] Num_encoder_layers { get => _num_encoder_layers; set => _num_encoder_layers = value; }
        public int TT { get => _TT; set => _TT = value; }
        public int[] Cnn_module_kernels { get => _cnn_module_kernels; set => _cnn_module_kernels = value; }
        public int[] Left_context_len { get => _left_context_len; set => _left_context_len = value; }
        public int Context_size { get => _context_size; set => _context_size = value; }
        public int Vocab_size { get => _vocab_size; set => _vocab_size = value; }
        public int Joiner_dim { get => _joiner_dim; set => _joiner_dim = value; }
        public int[] Query_head_dims { get => _query_head_dims; set => _query_head_dims = value; }
        public string? Onnx_infer { get => _onnx_infer; set => _onnx_infer = value; }
        public string? Comment { get => _comment; set => _comment = value; }
        public int[] Value_head_dims { get => _value_head_dims; set => _value_head_dims = value; }
        public int[] Num_heads { get => _num_heads; set => _num_heads = value; }
        public string Feature_type { get => feature_type; set => feature_type = value; }
        public int D_model { get => _d_model; set => _d_model = value; }
        public int Rnn_hidden_size { get => _rnn_hidden_size; set => _rnn_hidden_size = value; }
        public int Cnn_module_kernel { get => _cnn_module_kernel; set => _cnn_module_kernel = value; }
        public int Pad_length { get => _pad_length; set => _pad_length = value; }
        public int Encoder_dim { get => _encoder_dim; set => _encoder_dim = value; }
        public int Chunk_size { get => _chunk_size; set => _chunk_size = value; }
        public int Left_context { get => _left_context; set => _left_context = value; }
        public int Right_context { get => _right_context; set => _right_context = value; }
    }
}
