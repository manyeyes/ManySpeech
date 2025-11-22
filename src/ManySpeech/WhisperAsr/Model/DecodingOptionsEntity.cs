using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.WhisperAsr.Model
{
    public class DecodingOptionsEntity
    {
        // 是否执行X->X“转录”或X->英语“翻译”  
        public string Task { get; } = "transcribe";

        // 音频的语言；如果为null，则使用检测到的语言  
        public string? Language { get; }

        // 采样相关选项  
        public float Temperature { get; } = 0.0f;
        public int? SampleLen { get; } // 最大令牌数进行采样  
        public int? BestOf { get; } // 如果t > 0，则独立样本轨迹的数量  
        public int? BeamSize { get; } // 如果t == 0，则波束搜索中的波束数量  
        public float? Patience { get; } // 波束搜索中的耐心值（arxiv:2204.05424）  

        // 排名生成时使用的长度惩罚，类似于Google NMT中的“alpha”，或者如果为null，则使用长度归一化  
        public float? LengthPenalty { get; }

        // 作为提示或前缀的文本或令牌；更多信息：  
        // https://github.com/openai/whisper/discussions/117#discussioncomment-3727051  
        public List<int>? Prompt { get; } // 前一个上下文  
        public List<int>? Prefix { get; } // 当前上下文的前缀  

        // 要抑制的令牌ID列表（或逗号分隔的令牌ID）  
        // "-1"将抑制由tokenizer.non_speech_tokens()定义的一组符号  
        public int SuppressTokens { get; } // UnionType需要自定义或使用object和类型检查，或者使用string和List<int>  
        public bool SuppressBlank { get; } = true; // 这将抑制空白输出  

        // 时间戳采样选项  
        public bool WithoutTimestamps { get; } = false; // 使用<|notimestamps|>仅采样文本令牌  
        public float? MaxInitialTimestamp { get; } = 1.0f;

        // 由于C#构造函数不支持直接初始化只读字段，因此需要使用构造函数初始化它们  
        public DecodingOptionsEntity(
            string task = "transcribe",
            string? language = null,
            float temperature = 0.0f,
            int? sampleLen = null,
            int? bestOf = null,
            int? beamSize = null,
            float? patience = null,
            float? lengthPenalty = null,
            List<int>? prompt = null ,   
            List<int>? prefix = null,   
            int suppressTokens = -1,   
            bool suppressBlank = true,
            bool withoutTimestamps = false,
            float? maxInitialTimestamp = 1.0f)
        {
            Task = task;
            Language = language;
            Temperature = temperature;
            SampleLen = sampleLen;
            BestOf = bestOf;
            BeamSize = beamSize;
            Patience = patience;
            LengthPenalty = lengthPenalty;

            Prompt = prompt;
            Prefix = prefix;
            SuppressTokens = suppressTokens;

            SuppressBlank = suppressBlank;
            WithoutTimestamps = withoutTimestamps;
            MaxInitialTimestamp = maxInitialTimestamp;
        }
    }
}
