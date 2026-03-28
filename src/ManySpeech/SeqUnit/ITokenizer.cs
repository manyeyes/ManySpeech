namespace ManySpeech.SeqUnit
{
    /// <summary>
    /// Defines the interface for tokenizers that convert between text and token IDs.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Encodes the input text into an array of token IDs.
        /// </summary>
        /// <param name="text">The input text to encode.</param>
        /// <param name="isAllowSpecial">Specifies whether to allow special tokens during encoding. Default: false.</param>
        /// <returns>An array of token IDs; returns an empty array or null if the vocabulary is not loaded or the input text is empty (determined by the implementation).</returns>
        int[]? Encode(string text, bool isAllowSpecial = false);

        /// <summary>
        /// Decodes an array of token IDs into an array of individual token strings.
        /// </summary>
        /// <param name="tokenIds">The array of token IDs to decode.</param>
        /// <returns>An array of decoded token strings.</returns>
        string[] Decode(int[] tokenIds);

        /// <summary>
        /// Decodes an array of token IDs into a single complete text string.
        /// </summary>
        /// <param name="tokenIds">The array of token IDs to decode.</param>
        /// <returns>The complete decoded text string.</returns>
        string DecodeToText(int[] tokenIds);
    }
}