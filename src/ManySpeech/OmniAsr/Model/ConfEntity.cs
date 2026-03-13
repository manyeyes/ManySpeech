// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using YamlDotNet.Serialization;

namespace ManySpeech.OmniAsr.Model
{
    [YamlSerializable]
    public class ConfEntity
    {
        public int sampleRate { get; set; } = 16000;
        public string model { get; set; } = "omniasr-ctc";
        public string version { get; set; } = string.Empty;

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
        /// <param name="model">模型名称</param>
        /// <param name="version">版本</param>
        public ConfEntity(int sampleRate, string model,string version="0.1")
        {
            this.sampleRate = sampleRate;
            this.model = model;
            this.version = version;
        }
    }
}
