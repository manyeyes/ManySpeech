namespace ManySpeech.SeqUnit
{
    /// <summary>
    /// Tokenizer configuration entity.
    /// </summary>
    public class TokenizerConfigEntity
    {
        /// <summary>
        /// Tokenizer type (required).
        /// </summary>
        public TokenizerType Type { get; set; }

        /// <summary>
        /// Vocabulary file path (required).
        /// </summary>
        public string VocabPath { get; set; }

        /// <summary>
        /// Encoding name (only required for Tiktoken, optional, default: "multilingual").
        /// </summary>
        public string EncodingName { get; set; } = "multilingual";
    }

    /// <summary>
    /// Tokenizer configuration (read-only struct).
    /// </summary>
    public readonly struct TokenizerConfigStruct
    {
        /// <summary>
        /// Tokenizer type (required).
        /// </summary>
        public TokenizerType Type { get; }

        /// <summary>
        /// Vocabulary file path (required).
        /// </summary>
        public string VocabPath { get; }

        /// <summary>
        /// Encoding name (only required for Tiktoken, optional, default: "multilingual").
        /// </summary>
        public string EncodingName { get; }

        public TokenizerConfigStruct(TokenizerType type, string vocabPath, string encodingName = "multilingual")
        {
            Type = type;
            VocabPath = vocabPath ?? throw new ArgumentNullException(nameof(vocabPath));
            EncodingName = encodingName ?? throw new ArgumentNullException(nameof(encodingName));
        }
    }
}