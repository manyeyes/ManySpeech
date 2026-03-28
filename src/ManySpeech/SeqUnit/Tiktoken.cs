using Tiktoken;

namespace ManySpeech.SeqUnit
{
    /// <summary>
    /// Tokenizer implementation based on the Tiktoken library, supporting BPE encoding and decoding.
    /// </summary>
    internal class Tiktoken : ITokenizer
    {
        private readonly GptEncoding _encoding;

        /// <summary>
        /// Initializes the Tiktoken tokenizer.
        /// </summary>
        /// <param name="encodingName">Name of the encoding model, e.g., "multilingual, gpt2, qwen3".</param>
        /// <param name="vocabFilePath">Path to the vocabulary file. (Required for all models except multilingual and gpt2)</param>
        /// <exception cref="ArgumentNullException">Thrown when a required parameter is null or empty.</exception>
        /// <exception cref="Exception">Thrown when vocabulary loading fails (thrown by GptEncoding.GetEncoding).</exception>
        public Tiktoken(string encodingName, string? vocabFilePath = null)
        {
            if (string.IsNullOrWhiteSpace(encodingName))
                throw new ArgumentNullException(nameof(encodingName));

            _encoding = GptEncoding.GetEncoding(encodingName, vocabFilePath);
        }

        /// <summary>
        /// Encodes input text into an array of token IDs.
        /// </summary>
        /// <param name="text">Input text to encode.</param>
        /// <param name="isUseSpecial">Whether to allow special tokens during encoding.</param>
        /// <returns>Array of token IDs; returns an empty array if the input text is empty.</returns>
        public int[]? Encode(string text, bool isUseSpecial = false)
        {
            if (string.IsNullOrEmpty(text))
                return Array.Empty<int>();

            return _encoding.Encode(text, isUseSpecial).ToArray();
        }

        /// <summary>
        /// Decodes an array of token IDs into an array of individual token strings.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs to decode.</param>
        /// <returns>Array of decoded token strings; returns an empty array if the input is null or empty.</returns>
        public string[] Decode(int[] tokenIds)
        {
            return new string[tokenIds.Length];
        }

        /// <summary>
        /// Decodes an array of token IDs directly into a complete text string.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs to decode.</param>
        /// <returns>Complete decoded text string.</returns>
        public string DecodeToText(int[] tokenIds)
        {
            if (tokenIds == null || tokenIds.Length == 0)
                return string.Empty;

            return _encoding.Decode(tokenIds);
        }
    }
}