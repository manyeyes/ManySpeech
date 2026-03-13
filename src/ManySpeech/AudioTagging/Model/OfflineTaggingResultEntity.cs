// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
namespace ManySpeech.AudioTagging.Model
{
    /// <summary>
    /// 离线音频标注（Tagging）结果实体类
    /// 用于封装单次离线音频标注任务的所有输出结果，包含标签、token、概率、时间戳等核心信息
    /// </summary>
    public class OfflineTaggingResultEntity
    {
        /// <summary>
        /// 单标签标注结果（字符串形式）
        /// 场景：当音频为单类别分类时，存储最终的标注标签（如"音乐"、"汽车鸣笛"）
        /// </summary>
        private string? _tagging = null;

        /// <summary>
        /// 音频标注的Token索引列表
        /// 场景：存储模型输出的原始Token ID（对应标签词典的索引），用于后续映射为具体标签
        /// </summary>
        private List<int>? _tokens = new List<int>();

        /// <summary>
        /// 多标签标注结果列表
        /// 场景：当音频为多标签分类时，存储所有命中的标注标签（如["摇滚", "人声", "吉他"]）
        /// </summary>
        private List<string>? _taggings = new List<string>();

        /// <summary>
        /// 标注结果对应的概率值列表
        /// 场景：与Taggings/Tokens列表一一对应，存储每个标签/Token的置信度（0-1之间）
        /// </summary>
        private List<double> _probs = new List<double>();

        /// <summary>
        /// 标注结果的时间戳列表
        /// 场景：存储每个标注标签对应的音频时间区间，数组格式为 [起始时间(ms), 结束时间(ms)]
        /// 示例：[1000, 3000] 表示该标签对应音频1秒到3秒的区间
        /// </summary>
        private List<int[]>? _timestamps = new List<int[]>();

        /// <summary>
        /// 获取或设置音频标注的Token索引列表
        /// Token索引对应标签词典中的位置，可通过TokenConverter映射为具体标签
        /// </summary>
        public List<int>? Tokens { get => _tokens; set => _tokens = value; }

        /// <summary>
        /// 获取或设置标注结果的时间戳列表
        /// 每个元素为int[]数组，长度固定为2，分别表示起始时间和结束时间（单位：毫秒）
        /// </summary>
        public List<int[]>? Timestamps { get => _timestamps; set => _timestamps = value; }

        /// <summary>
        /// 获取或设置单标签标注结果
        /// 若为单类别分类场景，该字段存储最终的标注结果；多标签场景下该字段可为null
        /// </summary>
        public string? Tagging { get => _tagging; set => _tagging = value; }

        /// <summary>
        /// 获取或设置多标签标注结果列表
        /// 多标签分类场景下，存储所有命中的标签；单标签场景下该列表仅包含一个元素
        /// </summary>
        public List<string>? Taggings { get => _taggings; set => _taggings = value; }

        /// <summary>
        /// 获取或设置标注结果对应的概率值列表
        /// 概率值与Taggings列表元素一一对应，反映每个标签的置信度，值越大可信度越高
        /// </summary>
        public List<double> Probs { get => _probs; set => _probs = value; }
    }
}
