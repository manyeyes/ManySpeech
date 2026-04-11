// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using YamlDotNet.Serialization;

namespace ManySpeech.FireRedAsr.Model
{
    [YamlSerializable]
    /// <summary>
    /// 配置实体
    /// </summary>
    public class ConfEntity
    {
        ///// <summary>
        ///// 采样率
        ///// </summary>
        //public int SampleRate { get; set; } = 16000;

        /// <summary>
        /// 模型名称
        /// </summary>
        public string model { get; set; } = "fireredasraed_";

        /// <summary>
        /// 是否使用ITN
        /// </summary>
        public bool use_itn { get; set; } = false;

        /// <summary>
        /// 前端类型
        /// </summary>
        public string frontend { get; set; } = "wav_frontend";

        /// <summary>
        /// 前端配置
        /// </summary>
        public FrontendConf frontend_conf { get; set; } = new FrontendConf();

        /// <summary>
        /// 预处理器类型
        /// </summary>
        public string preprocessor { get; set; } = "s2t";

        /// <summary>
        /// 预处理器配置
        /// </summary>
        public PreprocessorConf preprocessor_conf { get; set; } = new PreprocessorConf();

        /// <summary>
        /// 版本号
        /// </summary>
        public string version { get; set; } = string.Empty;
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
        /// 特征类型
        /// </summary>
        private string _feature_type = "fbank";
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
        private int _lfr_m = 1;
        /// <summary>
        /// LFR的N参数
        /// </summary>
        private int _lfr_n = 1;
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

        private string _feature_concat_mode = "one_dimension";//one_dimension,matrix_column,frame_row

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
        public string feature_type { get => _feature_type; set => _feature_type = value; }
        public string feature_concat_mode { get => _feature_concat_mode; set => _feature_concat_mode = value; }
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

        public int pad_id { get; set; } = 3;

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
