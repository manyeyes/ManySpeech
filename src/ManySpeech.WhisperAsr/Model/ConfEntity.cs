namespace ManySpeech.WhisperAsr.Model
{
    public class DecodingOptions
    {
        // 是否执行X->X“转录”或X->英语“翻译”  
        public string? task { get; set; } = "transcribe";

        // 音频的语言；如果为null，则使用检测到的语言  
        public string? language { get; set; } = string.Empty;

        // 采样相关选项  
        public float temperature { get; set; } = 0.0f;
        public int? sample_len { get; set; } // 最大令牌数进行采样  
        public int? best_of { get; set; } // 如果t > 0，则独立样本轨迹的数量  
        public int? beam_size { get; set; } // 如果t == 0，则波束搜索中的波束数量  
        public float? patience { get; set; } // 波束搜索中的耐心值（arxiv:2204.05424）  

        // 排名生成时使用的长度惩罚，类似于Google NMT中的“alpha”，或者如果为null，则使用长度归一化  
        public float? length_penalty { get; set; }

        // 作为提示或前缀的文本或令牌；更多信息：  
        // https://github.com/openai/whisper/discussions/117#discussioncomment-3727051  
        public List<int>? prompt { get; set; } // 前一个上下文  
        public string? prompt_str { get; set; } = string.Empty; // 前一个上下文
        public List<int>? prefix { get; set; } // 当前上下文的前缀  
        public string? prefix_str { get; set; } = string.Empty; // 当前上下文的前缀 

        // 要抑制的令牌ID列表（或逗号分隔的令牌ID）  
        // "-1"将抑制由tokenizer.non_speech_tokens()定义的一组符号  
        public int[] suppress_tokens { get; set; } = new int[] { -1 };
        public string? suppress_tokens_str { get; set; } = string.Empty;
        public bool suppress_blank { get; set; } = true; // 这将抑制空白输出  

        // 时间戳采样选项  
        public bool without_timestamps { get; set; } = false; // 使用<|notimestamps|>仅采样文本令牌  
        public float? max_initial_timestamp { get; set; } = 1.0f;

        public bool fp16 { get; set; } = false;
    }

    public class ModelDimensions
    {
        private int _n_mels = 80;
        private int _n_audio_ctx = 1500;
        private int _n_audio_state = 768;
        private int _n_audio_head = 12;
        private int _n_audio_layer = 12;
        private int _n_vocab = 51865;
        private int _n_text_ctx = 448;
        private int _n_text_state = 768;
        private int _n_text_head = 12;
        private int _n_text_layer = 12;
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int n_audio_ctx { get => _n_audio_ctx; set => _n_audio_ctx = value; }
        public int n_audio_state { get => _n_audio_state; set => _n_audio_state = value; }
        public int n_audio_head { get => _n_audio_head; set => _n_audio_head = value; }
        public int n_audio_layer { get => _n_audio_layer; set => _n_audio_layer = value; }
        public int n_vocab { get => _n_vocab; set => _n_vocab = value; }
        public int n_text_ctx { get => _n_text_ctx; set => _n_text_ctx = value; }
        public int n_text_state { get => _n_text_state; set => _n_text_state = value; }
        public int n_text_head { get => _n_text_head; set => _n_text_head = value; }
        public int n_text_layer { get => _n_text_layer; set => _n_text_layer = value; }
    }
    public class AudioParameters
    {
        private int _n_mels = 80;
        private int _n_audio_ctx = 1500;
        private int _n_audio_state = 768;
        private int _n_audio_head = 12;
        private int _n_audio_layer = 12;
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int n_audio_ctx { get => _n_audio_ctx; set => _n_audio_ctx = value; }
        public int n_audio_state { get => _n_audio_state; set => _n_audio_state = value; }
        public int n_audio_head { get => _n_audio_head; set => _n_audio_head = value; }
        public int n_audio_layer { get => _n_audio_layer; set => _n_audio_layer = value; }
    }
    public class TextParameters
    {
        private int _n_vocab = 51865;
        private int _n_text_ctx = 448;
        private int _n_text_state = 768;
        private int _n_text_head = 12;
        private int _n_text_layer = 12;
        public int n_vocab { get => _n_vocab; set => _n_vocab = value; }
        public int n_text_ctx { get => _n_text_ctx; set => _n_text_ctx = value; }
        public int n_text_state { get => _n_text_state; set => _n_text_state = value; }
        public int n_text_head { get => _n_text_head; set => _n_text_head = value; }
        public int n_text_layer { get => _n_text_layer; set => _n_text_layer = value; }
    }
    public class ConfEntity
    {
        public bool is_multilingual { get; set; } = true;
        public int num_languages { get; set; } = 99;

        public ModelDimensions model_dimensions { get; set; } = new ModelDimensions();
        //public AudioParameters? audio_parameters { get; set; } = new AudioParameters();
        //public TextParameters? text_parameters { get; set; } = new TextParameters();
        public DecodingOptions? decoding_options { get; set; } = new DecodingOptions();
    }
}
