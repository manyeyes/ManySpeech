// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace ManySpeech.K2TransducerAsr.Model
{
    /// <summary>
    /// offline recognizer result entity 
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizerResultEntity
    {
        /// <summary>
        /// recognizer result
        /// </summary>
        public string? Text { get; set; }
        /// <summary>
        /// recognizer result length
        /// </summary>
        public int TextLen { get; set; }
        /// <summary>
        /// decode tokens
        /// </summary>
        public List<string>? Tokens { get; set; }

        /// <summary>
        /// timestamps
        /// </summary>
        public List<int[]>? Timestamps { get; set; }
    }
}
