// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.WenetAsr2
{
    /// <summary>
    /// online recognizer result entity 
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OnlineRecognizerResultEntity
    {
        /// <summary>
        /// recognizer result
        /// </summary>
        public string? text { get; set; }
        /// <summary>
        /// recognizer result length
        /// </summary>
        public int text_len { get; set; }
        /// <summary>
        /// decode tokens
        /// </summary>
        public List<string>? tokens { get; set; }
        /// <summary>
        /// timestamps
        /// </summary>
        public List<float>? timestamps { get; set; }
    }
}
