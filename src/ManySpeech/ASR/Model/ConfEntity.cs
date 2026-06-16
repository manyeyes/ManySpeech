// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using YamlDotNet.Serialization;

namespace ManySpeech.ASR.Model
{
    [YamlSerializable]
    public class ConfEntity
    {
        private string _model = "paraformer";
        private ModelConf _model_conf = new ModelConf();
        private string _encoder = "sanm";
        private EncoderConf _encoder_conf = new EncoderConf();
        private string _decoder = "paraformer_decoder_sanm";
        private DecoderConf _decoder_conf = new DecoderConf();
        private string _predictor = "cif_predictor_v2";
        private PredictorConf _predictor_conf = new PredictorConf();
        private bool _use_itn = false;
        private string _frontend = "wav_frontend";
        private FrontendConf _frontend_conf = new FrontendConf();
        private string _preprocessor = "s2t";
        private PreprocessorConf _preprocessor_conf = new PreprocessorConf();
        private string _version = string.Empty;

        public string model { get => _model; set => _model = value; }
        public ModelConf model_conf { get => _model_conf; set => _model_conf = value; }
        public string encoder { get => _encoder; set => _encoder = value; }
        public EncoderConf encoder_conf { get => _encoder_conf; set => _encoder_conf = value; }
        public string decoder { get => _decoder; set => _decoder = value; }
        public DecoderConf decoder_conf { get => _decoder_conf; set => _decoder_conf = value; }
        public string predictor { get => _predictor; set => _predictor = value; }
        public PredictorConf predictor_conf { get => _predictor_conf; set => _predictor_conf = value; }
        public bool use_itn { get => _use_itn; set => _use_itn = value; }
        public string frontend { get => _frontend; set => _frontend = value; }
        public FrontendConf frontend_conf { get => _frontend_conf; set => _frontend_conf = value; }
        public string preprocessor { get => _preprocessor; set => _preprocessor = value; }
        public PreprocessorConf preprocessor_conf { get => _preprocessor_conf; set => _preprocessor_conf = value; }
        public string version { get => _version; set => _version = value; }
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
    public class FrontendConf
    {
        /// <summary>
        /// 采样率 sample_rate
        /// </summary>
        private int _fs = 16000;
        /// <summary>
        /// 窗函数类型 window_type
        /// </summary>
        private string _window = "hamming";
        /// <summary>
        /// mel滤波器数量 num_bins
        /// </summary>
        private int _n_mels = 80;
        /// <summary>
        /// 帧长（单位：ms）frame_length
        /// </summary>
        private int _frame_length = 25;
        /// <summary>
        /// 帧移（单位：ms）frame_shift
        /// </summary>
        private int _frame_shift = 10;
        /// <summary>
        /// 抖动值 dither
        /// </summary>
        private float _dither = 1.0F;
        /// <summary>
        /// LFR的M参数
        /// </summary>
        private int _lfr_m = 7;
        /// <summary>
        /// LFR的N参数
        /// </summary>
        private int _lfr_n = 6;
        /// <summary>
        /// 是否裁剪边缘 snip_edges
        /// </summary>
        private bool _snip_edges = true;
        /// <summary>
        /// 是否使用librosa实现 is_librosa
        /// </summary>
        private bool _is_librosa = false;
        /// <summary>
        /// 是否启用HTK模式 htk_mode
        /// </summary>
        private bool _htk_mode = false;
        /// <summary>
        /// 最低频率 low_freq
        /// </summary>
        private float _low_freq = 20F;
        /// <summary>
        /// 最高频率 high_freq
        /// </summary>
        private float _high_freq = 0F;
        /// <summary>
        /// 归一化方式 norm
        /// </summary>
        private string _norm = "slaney";
        /// <summary>
        /// 是否移除直流偏移 remove_dc_offset
        /// </summary>
        private bool _remove_dc_offset = true;
        /// <summary>
        /// 预加重系数 preemph_coeff
        /// </summary>
        private float _preemph_coeff = 0.97f;
        /// <summary>
        /// 是否使用对数滤波器组 use_log_fbank
        /// </summary>
        private bool _use_log_fbank = true;
        /// <summary>
        /// active iff use_energy==true
        /// </summary>
        private float _energy_floor = 0.0f;

        public int fs { get => _fs; set => _fs = value; }
        public string window { get => _window; set => _window = value; }
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_shift { get => _frame_shift; set => _frame_shift = value; }
        public float dither { get => _dither; set => _dither = value; }
        public int lfr_m { get => _lfr_m; set => _lfr_m = value; }
        public int lfr_n { get => _lfr_n; set => _lfr_n = value; }
        public bool snip_edges { get => _snip_edges; set => _snip_edges = value; }
        public bool is_librosa { get => _is_librosa; set => _is_librosa = value; }
        public bool htk_mode { get => _htk_mode; set => _htk_mode = value; }
        public float low_freq { get => _low_freq; set => _low_freq = value; }
        public float high_freq { get => _high_freq; set => _high_freq = value; }
        public string norm { get => _norm; set => _norm = value; }
        public bool remove_dc_offset { get => _remove_dc_offset; set => _remove_dc_offset = value; }
        public float preemph_coeff { get => _preemph_coeff; set => _preemph_coeff = value; }
        public bool use_log_fbank { get => _use_log_fbank; set => _use_log_fbank = value; }
        public float energy_floor { get => _energy_floor; set => _energy_floor = value; }
    }
    /// <summary>
    /// 预处理器配置参数实体类
    /// </summary>
    public class PreprocessorConf
    {
        /// <summary>
        /// 前置文本名称
        /// </summary>
        public string text_prev_name { get; set; } = "text_prev";

        /// <summary>
        /// CTC 文本名称
        /// </summary>
        public string text_ctc_name { get; set; } = "text_ctc";

        /// <summary>
        /// 采样率（Hz）
        /// </summary>
        public int fs { get; set; } = 16000;

        /// <summary>
        /// NA 符号
        /// </summary>
        public string na_symbol { get; set; } = "<na>";

        /// <summary>
        /// 语音长度（秒）
        /// </summary>
        public int speech_length { get; set; } = 30;

        /// <summary>
        /// 语音分辨率（秒）
        /// </summary>
        public float speech_resolution { get; set; } = 0.02f;

        /// <summary>
        /// 语音初始静音长度
        /// </summary>
        public int speech_init_silence { get; set; } = 30;

        /// <summary>
        /// 前置文本应用概率
        /// </summary>
        public float text_prev_apply_prob { get; set; } = 0.3f;

        /// <summary>
        /// 时间戳应用概率
        /// </summary>
        public float time_apply_prob { get; set; } = 0.5f;

        /// <summary>
        /// 无时间戳符号
        /// </summary>
        public string notime_symbol { get; set; } = "<notimestamp>";

        /// <summary>
        /// 首个时间戳符号
        /// </summary>
        public string first_time_symbol { get; set; } = "<0.00>";

        /// <summary>
        /// 最后一个时间戳符号
        /// </summary>
        public string last_time_symbol { get; set; } = "<30.00>";

        /// <summary>
        /// 空白符ID <blank>
        /// </summary>
        public int blank_id { get; set; } = 0;

        /// <summary>
        /// 开始/结束符共用ID <sos/eos>
        /// </summary>
        public int sos_eos_id { get; set; } = 1;

        /// <summary>
        /// 未知字符ID <unk>
        /// </summary>
        public int unk_id { get; set; } = 2;

        /// <summary>
        /// 语言ID起始 <ab>
        /// </summary>
        public int first_lang_id { get; set; } = 7;

        /// <summary>
        /// 语言ID结束 <zu>
        /// </summary>
        public int last_lang_id { get; set; } = 144;

        /// <summary>
        /// 地区ID起始 <AD>
        /// </summary>
        public int first_region_id { get; set; } = 145;

        /// <summary>
        /// 地区ID结束 <ZA>
        /// </summary>
        public int last_region_id { get; set; } = 323;

        /// <summary>
        /// 序列开始符ID <sos>
        /// </summary>
        public int sos_id { get; set; } = 39999;

        /// <summary>
        /// 序列结束符ID <eos>
        /// </summary>
        public int eos_id { get; set; } = 40000;

        /// <summary>
        /// ASR任务标识ID <asr>
        /// </summary>
        public int asr_id { get; set; } = 6;

        /// <summary>
        /// 是否对音频进行长度统一裁剪 / 补齐
        /// </summary>
        public bool is_resize_audio_duration { get; set; } = false;

        /// <summary>
        /// 是否对语音进行 padding
        /// </summary>
        public bool is_padding_speech { get; set; } = false;

        public bool is_sample_scaling_required { get; set; } = false;

        /// <summary>
        /// 是否对批次语音进行 padding
        /// </summary>
        public bool batch_padding_speech { get; set; } = true;

        /// <summary>
        /// 批次语音 padding 概率
        /// </summary>
        public float batch_padding_speech_prob { get; set; } = 0.5f;

        /// <summary>
        /// 是否使用外部前端
        /// </summary>
        public bool use_wavfrontend { get; set; } = true;
    }
}
