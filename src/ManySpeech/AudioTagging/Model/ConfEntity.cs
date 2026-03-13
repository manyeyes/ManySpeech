// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
namespace ManySpeech.AudioTagging.Model
{
    /// <summary>
    /// 音频分类/标注的核心参数配置类
    /// 包含采样率、mel频谱维度、帧移等关键参数
    /// </summary>
    public class ConfEntity
    {
        /// <summary>
        /// 音频采样率（赫兹），默认值 16000
        /// </summary>
        public int sampleRate { get; set; } = 16000;

        /// <summary>
        /// Mel频谱的维度（特征数量），默认值 64
        /// </summary>
        public int melSpectrumDimension { get; set; } = 64;

        /// <summary>
        /// 帧移（每次滑动的样本数），默认值 160
        /// </summary>
        public int hopLength { get; set; } = 160;

        /// <summary>
        /// 模型期望的时间帧数（输入特征的时间维度长度），默认值 1012
        /// </summary>
        public int targetTimeFrameCount { get; set; } = 1012;

        /// <summary>
        /// 音频分块长度（秒），默认值 10.0
        /// </summary>
        public double audioChunkLength { get; set; } = 10.0;

        public string model { get; set; } = "ced";

        /// <summary>
        /// 无参构造函数，使用默认参数值
        /// </summary>
        public ConfEntity()
        {
        }

        /// <summary>
        /// 带参构造函数，支持自定义所有参数
        /// </summary>
        /// <param name="sampleRate">采样率</param>
        /// <param name="melSpectrumDimension">mel频谱维度</param>
        /// <param name="hopLength">帧移</param>
        /// <param name="targetTimeFrameCount">目标时间帧数</param>
        /// <param name="audioChunkLength">音频分块长度（秒）</param>
        public ConfEntity(int sampleRate, int melSpectrumDimension, int hopLength, int targetTimeFrameCount, double audioChunkLength, string model)
        {
            this.sampleRate = sampleRate;
            this.melSpectrumDimension = melSpectrumDimension;
            this.hopLength = hopLength;
            this.targetTimeFrameCount = targetTimeFrameCount;
            this.audioChunkLength = audioChunkLength;
            this.model = model;
        }
    }
}
