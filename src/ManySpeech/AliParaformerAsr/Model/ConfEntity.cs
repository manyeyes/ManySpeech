// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using YamlDotNet.Serialization;

namespace ManySpeech.AliParaformerAsr.Model
{
    [YamlSerializable]
    public class ConfEntity
    {
        private int _input_size;
        private string _frontend = "wav_frontend";
        private FrontendConf _frontend_conf = new FrontendConf();
        private string _model = "paraformer";
        private bool _use_itn = false;
        private ModelConf _model_conf = new ModelConf();
        private string _preencoder = string.Empty;
        //private PreEncoderConf _preencoder_conf = new PreEncoderConf();
        private string _encoder = "sanm";
        private EncoderConf _encoder_conf = new EncoderConf();
        private string _postencoder = string.Empty;
        //private PostEncoderConf _postencoder_conf = new PostEncoderConf();
        private string _decoder = "paraformer_decoder_sanm";
        private DecoderConf _decoder_conf = new DecoderConf();
        private string _predictor = "cif_predictor_v2";
        private PredictorConf _predictor_conf = new PredictorConf();
        private string _version = string.Empty;


        public int input_size { get => _input_size; set => _input_size = value; }
        public string frontend { get => _frontend; set => _frontend = value; }
        public FrontendConf frontend_conf { get => _frontend_conf; set => _frontend_conf = value; }
        public string model { get => _model; set => _model = value; }
        public ModelConf model_conf { get => _model_conf; set => _model_conf = value; }
        public string preencoder { get => _preencoder; set => _preencoder = value; }
        //public PreEncoderConf preencoder_conf { get => _preencoder_conf; set => _preencoder_conf = value; }
        public string encoder { get => _encoder; set => _encoder = value; }
        public EncoderConf encoder_conf { get => _encoder_conf; set => _encoder_conf = value; }
        public string postencoder { get => _postencoder; set => _postencoder = value; }
        //public PostEncoderConf postencoder_conf { get => _postencoder_conf; set => _postencoder_conf = value; }
        public string decoder { get => _decoder; set => _decoder = value; }
        public DecoderConf decoder_conf { get => _decoder_conf; set => _decoder_conf = value; }
        public string predictor { get => _predictor; set => _predictor = value; }
        public string version { get => _version; set => _version = value; }
        public PredictorConf predictor_conf { get => _predictor_conf; set => _predictor_conf = value; }
        public bool use_itn { get => _use_itn; set => _use_itn = value; }
    }
    public class FrontendConf
    {
        private int _fs = 16000;
        private string _window = "hamming";
        private int _n_mels = 80;
        private int _frame_length = 25;
        private int _frame_shift = 10;
        private float _dither = 1.0F;
        private int _lfr_m = 7;
        private int _lfr_n = 6;
        private bool _snip_edges = false;

        public int fs { get => _fs; set => _fs = value; }
        public string window { get => _window; set => _window = value; }
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_shift { get => _frame_shift; set => _frame_shift = value; }
        public float dither { get => _dither; set => _dither = value; }
        public int lfr_m { get => _lfr_m; set => _lfr_m = value; }
        public int lfr_n { get => _lfr_n; set => _lfr_n = value; }
        public bool snip_edges { get => _snip_edges; set => _snip_edges = value; }
    }
    public class ModelConf
    {
        private float _ctc_weight = 0.0F;
        private float _lsm_weight = 0.1F;
        private bool _length_normalized_loss = true;
        private float _predictor_weight = 1.0F;
        private int _predictor_bias = 1;
        private float _sampling_ratio = 0.75F;
        private int _sos = 1;
        private int _eos = 2;
        private int _ignore_id = -1;

        public float ctc_weight { get => _ctc_weight; set => _ctc_weight = value; }
        public float lsm_weight { get => _lsm_weight; set => _lsm_weight = value; }
        public bool length_normalized_loss { get => _length_normalized_loss; set => _length_normalized_loss = value; }
        public float predictor_weight { get => _predictor_weight; set => _predictor_weight = value; }
        public int predictor_bias { get => _predictor_bias; set => _predictor_bias = value; }
        public float sampling_ratio { get => _sampling_ratio; set => _sampling_ratio = value; }
        public int sos { get => _sos; set => _sos = value; }
        public int eos { get => _eos; set => _eos = value; }
        public int ignore_id { get => _ignore_id; set => _ignore_id = value; }
    }
    //public class PreEncoderConf
    //{
    //}
    public class EncoderConf
    {
        private int _output_size = 512;
        private int _attention_heads = 4;
        private int _linear_units = 2048;
        private int _num_blocks = 50;
        private float _dropout_rate = 0.1F;
        private float _positional_dropout_rate = 0.1F;
        private float _attention_dropout_rate = 0.1F;
        private string _input_layer = "pe";
        private string _pos_enc_class = "SinusoidalPositionEncoder";
        private bool _normalize_before = true;
        private int _kernel_size = 11;
        private int _sanm_shfit = 0;
        private string _selfattention_layer_type = "sanm";

        public int output_size { get => _output_size; set => _output_size = value; }
        public int attention_heads { get => _attention_heads; set => _attention_heads = value; }
        public int linear_units { get => _linear_units; set => _linear_units = value; }
        public int num_blocks { get => _num_blocks; set => _num_blocks = value; }
        public float dropout_rate { get => _dropout_rate; set => _dropout_rate = value; }
        public float positional_dropout_rate { get => _positional_dropout_rate; set => _positional_dropout_rate = value; }
        public float attention_dropout_rate { get => _attention_dropout_rate; set => _attention_dropout_rate = value; }
        public string input_layer { get => _input_layer; set => _input_layer = value; }
        public string pos_enc_class { get => _pos_enc_class; set => _pos_enc_class = value; }
        public bool normalize_before { get => _normalize_before; set => _normalize_before = value; }
        public int kernel_size { get => _kernel_size; set => _kernel_size = value; }
        public int sanm_shfit { get => _sanm_shfit; set => _sanm_shfit = value; }
        public string selfattention_layer_type { get => _selfattention_layer_type; set => _selfattention_layer_type = value; }
    }
    //public class PostEncoderConf
    //{
    //}
    public class DecoderConf
    {
        private int _attention_heads = 4;
        private int _linear_units = 2048;
        private int _num_blocks = 16;
        private float _dropout_rate = 0.1F;
        private float _positional_dropout_rate = 0.1F;
        private float _self_attention_dropout_rate = 0.1F;
        private float _src_attention_dropout_rate = 0.1F;
        private int _att_layer_num = 16;
        private int _kernel_size = 11;
        private int _sanm_shfit = 0;

        public int attention_heads { get => _attention_heads; set => _attention_heads = value; }
        public int linear_units { get => _linear_units; set => _linear_units = value; }
        public int num_blocks { get => _num_blocks; set => _num_blocks = value; }
        public float dropout_rate { get => _dropout_rate; set => _dropout_rate = value; }
        public float positional_dropout_rate { get => _positional_dropout_rate; set => _positional_dropout_rate = value; }
        public float self_attention_dropout_rate { get => _self_attention_dropout_rate; set => _self_attention_dropout_rate = value; }
        public float src_attention_dropout_rate { get => _src_attention_dropout_rate; set => _src_attention_dropout_rate = value; }
        public int att_layer_num { get => _att_layer_num; set => _att_layer_num = value; }
        public int kernel_size { get => _kernel_size; set => _kernel_size = value; }
        public int sanm_shfit { get => _sanm_shfit; set => _sanm_shfit = value; }

    }
    public class PredictorConf
    {
        private int _idim = 512;
        private float _threshold = 1.0F;
        private int _l_order = 1;
        private int _r_order = 1;
        private float _tail_threshold = 0.45F;

        public int idim { get => _idim; set => _idim = value; }
        public float threshold { get => _threshold; set => _threshold = value; }
        public int l_order { get => _l_order; set => _l_order = value; }
        public int r_order { get => _r_order; set => _r_order = value; }
        public float tail_threshold { get => _tail_threshold; set => _tail_threshold = value; }
    }
}
