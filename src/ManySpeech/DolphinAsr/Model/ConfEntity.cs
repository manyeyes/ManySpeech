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
    }
}

