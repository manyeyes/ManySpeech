using YamlDotNet.Serialization;

namespace ManySpeech.DolphinAsr.Model
{
    [YamlSerializable]
    public class ConfEntity
    {
        /// <summary>
        /// 编码器类型（如 e_branchformer）
        /// </summary>
        public string encoder { get; set; }

        /// <summary>
        /// 编码器配置参数
        /// </summary>
        public EncoderConfig encoder_conf { get; set; }

        /// <summary>
        /// 解码器类型（如 transformer）
        /// </summary>
        public string decoder { get; set; }

        /// <summary>
        /// 解码器配置参数
        /// </summary>
        public DecoderConfig decoder_conf { get; set; }

        /// <summary>
        /// 预处理器类型（如 s2t）
        /// </summary>
        public string preprocessor { get; set; }

        /// <summary>
        /// 预处理器配置参数
        /// </summary>
        public PreprocessorConfig preprocessor_conf { get; set; }
        public FrontendConfig frontend_conf { get; set; }

        /// <summary>
        /// 初始化默认模型配置（加载预设参数）
        /// </summary>
        /// <returns>初始化完成的 ConfEntity 实例</returns>
        public static ConfEntity InitializeDefault()
        {
            return new ConfEntity
            {
                encoder = "e_branchformer",
                encoder_conf = new EncoderConfig
                {
                    output_size = 512,
                    attention_heads = 8,
                    attention_layer_type = "rel_selfattn",
                    pos_enc_layer_type = "rel_pos",
                    rel_pos_type = "latest",
                    cgmlp_linear_units = 2048,
                    cgmlp_conv_kernel = 31,
                    use_linear_after_conv = false,
                    gate_activation = "identity",
                    num_blocks = 6,
                    dropout_rate = 0.1f,
                    positional_dropout_rate = 0.1f,
                    attention_dropout_rate = 0.1f,
                    input_layer = "conv2d",
                    layer_drop_rate = 0.0f,
                    linear_units = 2048,
                    positionwise_layer_type = "linear",
                    use_ffn = true,
                    macaron_ffn = true,
                    merge_conv_kernel = 31
                },
                decoder = "transformer",
                decoder_conf = new DecoderConfig
                {
                    attention_heads = 8,
                    linear_units = 2048,
                    num_blocks = 6,
                    dropout_rate = 0.1f,
                    positional_dropout_rate = 0.1f,
                    self_attention_dropout_rate = 0.1f,
                    src_attention_dropout_rate = 0.1f
                },
                preprocessor = "s2t",
                preprocessor_conf = new PreprocessorConfig
                {
                    text_prev_name = "text_prev",
                    text_ctc_name = "text_ctc",
                    fs = 16000,
                    na_symbol = "<na>",
                    speech_length = 30,
                    speech_resolution = 0.02f,
                    speech_init_silence = 30,
                    text_prev_apply_prob = 0.3f,
                    time_apply_prob = 0.5f,
                    notime_symbol = "<notimestamp>",
                    first_time_symbol = "<0.00>",
                    last_time_symbol = "<30.00>",
                    is_padding_speech = false,
                    batch_padding_speech = true,
                    batch_padding_speech_prob = 0.5f
                },
                frontend_conf=new FrontendConfig
                {
                    fs = 16000,
                    window = "hanning",
                    n_mels = 80,
                    frame_length = 32,
                    frame_shift = 10,
                    dither = 0F,
                    lfr_m = 7,
                    lfr_n = 6,
                    snip_edges = false,
                    is_librosa = false,
                    htk_mode = false,
                    low_freq = 0F,
                    high_freq = 8000F,
                    norm = "",
                    remove_dc_offset = false,
                    preemph_coeff = 0f,
                    use_log_fbank = true
                }
            };
        }
    }
    /// <summary>
    /// 编码器配置参数实体类（属性名与JSON键名完全一致）
    /// </summary>
    public class EncoderConfig
    {
        /// <summary>
        /// 编码器输出特征维度（核心参数）
        /// </summary>
        public int output_size { get; set; }

        /// <summary>
        /// 注意力头数
        /// </summary>
        public int attention_heads { get; set; }

        /// <summary>
        /// 注意力层类型
        /// </summary>
        public string attention_layer_type { get; set; }

        /// <summary>
        /// 位置编码层类型
        /// </summary>
        public string pos_enc_layer_type { get; set; }

        /// <summary>
        /// 相对位置编码类型
        /// </summary>
        public string rel_pos_type { get; set; }

        /// <summary>
        /// CGMLP 线性单元数
        /// </summary>
        public int cgmlp_linear_units { get; set; }

        /// <summary>
        /// CGMLP 卷积核大小
        /// </summary>
        public int cgmlp_conv_kernel { get; set; }

        /// <summary>
        /// 卷积后是否使用线性层
        /// </summary>
        public bool use_linear_after_conv { get; set; }

        /// <summary>
        /// 门控激活函数类型
        /// </summary>
        public string gate_activation { get; set; }

        /// <summary>
        /// 编码器块数量（层数）
        /// </summary>
        public int num_blocks { get; set; }

        /// <summary>
        /// 通用 dropout 率
        /// </summary>
        public float dropout_rate { get; set; }

        /// <summary>
        /// 位置编码 dropout 率
        /// </summary>
        public float positional_dropout_rate { get; set; }

        /// <summary>
        /// 注意力层 dropout 率
        /// </summary>
        public float attention_dropout_rate { get; set; }

        /// <summary>
        /// 输入层类型
        /// </summary>
        public string input_layer { get; set; }

        /// <summary>
        /// 层 dropout 率
        /// </summary>
        public float layer_drop_rate { get; set; }

        /// <summary>
        /// 位置感知前馈网络线性单元数
        /// </summary>
        public int linear_units { get; set; }

        /// <summary>
        /// 位置感知层类型
        /// </summary>
        public string positionwise_layer_type { get; set; }

        /// <summary>
        /// 是否使用前馈网络
        /// </summary>
        public bool use_ffn { get; set; }

        /// <summary>
        /// 是否使用 Macaron 风格 FFN
        /// </summary>
        public bool macaron_ffn { get; set; }

        /// <summary>
        /// 合并卷积核大小
        /// </summary>
        public int merge_conv_kernel { get; set; }
    }

    /// <summary>
    /// 解码器配置参数实体类（属性名与JSON键名完全一致）
    /// </summary>
    public class DecoderConfig
    {
        /// <summary>
        /// 注意力头数
        /// </summary>
        public int attention_heads { get; set; }

        /// <summary>
        /// 线性单元数
        /// </summary>
        public int linear_units { get; set; }

        /// <summary>
        /// 解码器块数量（层数）
        /// </summary>
        public int num_blocks { get; set; }

        /// <summary>
        /// 通用 dropout 率
        /// </summary>
        public float dropout_rate { get; set; }

        /// <summary>
        /// 位置编码 dropout 率
        /// </summary>
        public float positional_dropout_rate { get; set; }

        /// <summary>
        /// 自注意力层 dropout 率
        /// </summary>
        public float self_attention_dropout_rate { get; set; }

        /// <summary>
        /// 源注意力层 dropout 率
        /// </summary>
        public float src_attention_dropout_rate { get; set; }
    }

    /// <summary>
    /// 预处理器配置参数实体类（属性名与JSON键名完全一致）
    /// </summary>
    public class PreprocessorConfig
    {
        /// <summary>
        /// 前置文本名称
        /// </summary>
        public string text_prev_name { get; set; }

        /// <summary>
        /// CTC 文本名称
        /// </summary>
        public string text_ctc_name { get; set; }

        /// <summary>
        /// 采样率（Hz）
        /// </summary>
        public int fs { get; set; }

        /// <summary>
        /// NA 符号
        /// </summary>
        public string na_symbol { get; set; }

        /// <summary>
        /// 语音长度（秒）
        /// </summary>
        public int speech_length { get; set; }

        /// <summary>
        /// 语音分辨率（秒）
        /// </summary>
        public float speech_resolution { get; set; }

        /// <summary>
        /// 语音初始静音长度
        /// </summary>
        public int speech_init_silence { get; set; }

        /// <summary>
        /// 前置文本应用概率
        /// </summary>
        public float text_prev_apply_prob { get; set; }

        /// <summary>
        /// 时间戳应用概率
        /// </summary>
        public float time_apply_prob { get; set; }

        /// <summary>
        /// 无时间戳符号
        /// </summary>
        public string notime_symbol { get; set; }

        /// <summary>
        /// 首个时间戳符号
        /// </summary>
        public string first_time_symbol { get; set; }

        /// <summary>
        /// 最后一个时间戳符号
        /// </summary>
        public string last_time_symbol { get; set; }

        /// <summary>
        /// 是否对语音进行 padding
        /// </summary>
        public bool is_padding_speech { get; set; }

        /// <summary>
        /// 是否对批次语音进行 padding
        /// </summary>
        public bool batch_padding_speech { get; set; }

        /// <summary>
        /// 批次语音 padding 概率
        /// </summary>
        public float batch_padding_speech_prob { get; set; }

        /// <summary>
        /// 是否使用外部前端
        /// </summary>
        public bool use_wavfrontend { get; set; }
    }
    /// <summary>
    /// 前端配置实体类（Frontend Configuration）
    /// </summary>
    public class FrontendConfig
    {
        /// <summary>
        /// 采样率
        /// </summary>
        private int _fs = 16000;
        /// <summary>
        /// 窗函数类型（如hanning）
        /// </summary>
        private string _window = "hanning";
        /// <summary>
        /// mel滤波器数量
        /// </summary>
        private int _n_mels = 80;
        /// <summary>
        /// 帧长（单位：ms）
        /// </summary>
        private int _frame_length = 32;
        /// <summary>
        /// 帧移（单位：ms）
        /// </summary>
        private int _frame_shift = 10;
        /// <summary>
        /// 抖动值
        /// </summary>
        private float _dither = 0F;
        /// <summary>
        /// LFR的M参数
        /// </summary>
        private int _lfr_m = 7;
        /// <summary>
        /// LFR的N参数
        /// </summary>
        private int _lfr_n = 6;
        /// <summary>
        /// 是否裁剪边缘
        /// </summary>
        private bool _snip_edges = false;
        /// <summary>
        /// 是否使用librosa实现
        /// </summary>
        private bool _is_librosa = false;
        /// <summary>
        /// 是否启用HTK模式
        /// </summary>
        private bool _htk_mode = false;
        /// <summary>
        /// 最低频率
        /// </summary>
        private float _low_freq = 0F;
        /// <summary>
        /// 最高频率
        /// </summary>
        private float _high_freq = 8000F;
        /// <summary>
        /// 归一化方式
        /// </summary>
        private string _norm = "";
        /// <summary>
        /// 是否移除直流偏移
        /// </summary>
        private bool _remove_dc_offset = false;
        /// <summary>
        /// 预加重系数
        /// </summary>
        private float _preemph_coeff = 0f;
        /// <summary>
        /// 是否使用对数滤波器组
        /// </summary>
        private bool _use_log_fbank = true;

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
    }
}

