namespace ManySpeech.SeqUnit
{
    /// <summary>
    /// Tokenizer type enumeration.
    /// </summary>
    public enum TokenizerType
    {
        /// <summary>
        /// Vocabulary-based tokenizer (Textoken)
        /// </summary>
        Textoken,

        /// <summary>
        /// BPE-based tokenizer using Tiktoken
        /// </summary>
        Tiktoken
    }

    /// <summary>
    /// Auto tokenizer factory that returns the corresponding ITokenizer implementation based on configuration.
    /// </summary>
    internal class AutoTokenizer
    {
        /// <summary>
        /// Creates a tokenizer instance based on the specified type and parameters.
        /// </summary>
        /// <param name="type">Tokenizer type.</param>
        /// <param name="vocabFilePath">Vocabulary file path (required for both Textoken and Tiktoken).</param>
        /// <param name="encodingName">Encoding name (only required for Tiktoken, default is "multilingual").</param>
        /// <returns>ITokenizer instance.</returns>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
        public static ITokenizer Create(TokenizerType type, string vocabFilePath, string encodingName = "multilingual")
        {
            if (string.IsNullOrWhiteSpace(vocabFilePath))
                throw new ArgumentException("Vocabulary file path cannot be empty.", nameof(vocabFilePath));

            return type switch
            {
                TokenizerType.Textoken => new Textoken(vocabFilePath),
                TokenizerType.Tiktoken => new Tiktoken(encodingName, vocabFilePath),
                _ => throw new NotSupportedException($"Unsupported tokenizer type: {type}")
            };
        }

        /// <summary>
        /// Creates a tokenizer instance based on a configuration dictionary.
        /// The dictionary must contain keys:
        /// - "type": TokenizerType or string "textoken"/"tiktoken"
        /// - "vocabPath": Vocabulary file path (required)
        /// - "encodingName": Encoding name (optional, only for Tiktoken)
        /// </summary>
        /// <param name="config">Configuration dictionary.</param>
        /// <returns>ITokenizer instance.</returns>
        public static ITokenizer Create(Dictionary<string, object> config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (!config.TryGetValue("type", out object? typeObj))
                throw new ArgumentException("Missing 'type' key in configuration.");

            if (!config.TryGetValue("vocabPath", out object? vocabObj) || vocabObj == null)
                throw new ArgumentException("Missing 'vocabPath' key in configuration.");

            string? vocabPath = vocabObj.ToString();
            if (string.IsNullOrWhiteSpace(vocabPath))
                throw new ArgumentException("vocabPath cannot be empty.");

            // Parse tokenizer type
            TokenizerType type = typeObj switch
            {
                TokenizerType t => t,
                string s when s.Equals("textoken", StringComparison.OrdinalIgnoreCase) => TokenizerType.Textoken,
                string s when s.Equals("tiktoken", StringComparison.OrdinalIgnoreCase) => TokenizerType.Tiktoken,
                _ => throw new ArgumentException($"Invalid tokenizer type: {typeObj}")
            };

            string? encodingName = config.TryGetValue("encodingName", out object? encObj) && encObj != null
                ? encObj.ToString()
                : "multilingual";

            // Delegate to the main overload
            return Create(type, vocabPath!, encodingName!);
        }

        /// <summary>
        /// Creates a tokenizer instance based on a configuration entity (class).
        /// </summary>
        /// <param name="config">Tokenizer configuration entity.</param>
        /// <returns>ITokenizer instance.</returns>
        public static ITokenizer Create(TokenizerConfigEntity config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (string.IsNullOrWhiteSpace(config.VocabPath))
                throw new ArgumentException("VocabPath cannot be empty.", nameof(config.VocabPath));

            // Delegate to the main overload
            return Create(config.Type, config.VocabPath, config.EncodingName);
        }

        /// <summary>
        /// Creates a tokenizer instance based on a configuration structure (struct).
        /// </summary>
        /// <param name="config">Tokenizer configuration struct.</param>
        /// <returns>ITokenizer instance.</returns>
        public static ITokenizer Create(TokenizerConfigStruct config)
        {
            if (string.IsNullOrWhiteSpace(config.VocabPath))
                throw new ArgumentException("VocabPath cannot be empty.", nameof(config));

            // Delegate to the main overload
            return Create(config.Type, config.VocabPath, config.EncodingName);
        }
    }
}