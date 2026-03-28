using System.Reflection;

namespace ManySpeech.SeqUnit
{
    /// <summary>
    /// Vocabulary-based text tokenizer supporting encoding (text → token IDs) and decoding (token IDs → text).
    /// </summary>
    internal class Textoken : ITokenizer
    {
        private readonly string[]? _tokens;                 // Raw token lines, format: "token\tid" or pure token
        private readonly Dictionary<string, int> _tokenToId; // Token text → ID mapping
        private readonly List<string> _sortedTokens;        // Token list sorted by length descending for maximum matching

        /// <summary>
        /// Initializes the tokenizer without loading a vocabulary.
        /// </summary>
        public Textoken()
        {
            _tokens = null;
            _tokenToId = new Dictionary<string, int>();
            _sortedTokens = new List<string>();
        }

        /// <summary>
        /// Loads vocabulary from the specified path or embedded resource.
        /// </summary>
        /// <param name="tokensFilePath">Vocabulary file path. If it is a filename (no path separators), it is treated as an embedded resource name.</param>
        /// <exception cref="FileNotFoundException">Thrown when the file or embedded resource does not exist.</exception>
        public Textoken(string tokensFilePath)
        {
            if (string.IsNullOrWhiteSpace(tokensFilePath))
                throw new ArgumentException("Vocabulary file path cannot be empty", nameof(tokensFilePath));

            _tokens = ReadTokens(tokensFilePath);
            (_tokenToId, _sortedTokens) = BuildTokenMappings(_tokens);
        }

        /// <summary>
        /// Encodes input text into an array of token IDs.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <param name="isUseSpecial">Whether to use special tokens (reserved for future use).</param>
        /// <returns>Array of token IDs; returns null or empty array if vocabulary is not loaded or text is empty.</returns>
        public int[]? Encode(string text, bool isUseSpecial = false)
        {
            if (_tokens == null || _tokens.Length == 0)
                return null; // Vocabulary not loaded

            if (string.IsNullOrEmpty(text))
                return Array.Empty<int>();

            string[] tokenStrings = GetTokens(text);
            var ids = new List<int>(tokenStrings.Length);
            foreach (var token in tokenStrings)
            {
                if (_tokenToId.TryGetValue(token, out int id))
                    ids.Add(id);
                else
                {
                    // Can be extended to log OOV tokens or throw exceptions
                }
            }
            return ids.ToArray();
        }

        /// <summary>
        /// Decodes an array of token IDs into an array of token strings.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs.</param>
        /// <returns>Decoded token string array; returns empty array if vocabulary is not loaded or IDs are invalid.</returns>
        public string[] Decode(int[] tokenIds)
        {
            if (_tokens == null || tokenIds == null || tokenIds.Length == 0)
                return Array.Empty<string>();

            var tokens = new List<string>(tokenIds.Length);
            foreach (int id in tokenIds)
            {
                if (id >= 0 && id < _tokens.Length)
                {
                    // Take only the first part as token text (supports "token\tid" format)
                    string token = _tokens[id].Split('\t')[0];
                    tokens.Add(token);
                }
                // Skip invalid IDs
            }
            return tokens.ToArray();
        }

        /// <summary>
        /// Decodes an array of token IDs directly into a complete string.
        /// </summary>
        /// <param name="tokenIds">Array of token IDs.</param>
        /// <returns>Complete decoded text string.</returns>
        public string DecodeToText(int[] tokenIds)
        {
            if (_tokens == null || tokenIds == null || tokenIds.Length == 0)
                return string.Empty;

            string[] tokens = Decode(tokenIds);
            return string.Concat(tokens);
        }

        /// <summary>
        /// Reads vocabulary from a file or embedded resource, one token per line (supports "token\tid" format).
        /// </summary>
        /// <param name="tokensFilePath">File path or embedded resource name.</param>
        /// <returns>Token string array.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the file or resource does not exist.</exception>
        public static string[] ReadTokens(string tokensFilePath)
        {
            if (string.IsNullOrEmpty(tokensFilePath))
                throw new ArgumentException("Path cannot be empty", nameof(tokensFilePath));

            // Treat as embedded resource if no path separators
            bool isEmbedded = tokensFilePath.IndexOf('/') < 0 && tokensFilePath.IndexOf('\\') < 0;

            if (isEmbedded)
            {
                var assembly = Assembly.GetExecutingAssembly();
                using var stream = assembly.GetManifestResourceStream(tokensFilePath)
                    ?? throw new FileNotFoundException($"Embedded resource '{tokensFilePath}' not found.");
                using var reader = new StreamReader(stream);
                return reader.ReadToEnd().Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            }
            else
            {
                if (!File.Exists(tokensFilePath))
                    throw new FileNotFoundException($"File '{tokensFilePath}' does not exist.");
                return File.ReadAllLines(tokensFilePath);
            }
        }

        /// <summary>
        /// Performs forward maximum matching tokenization on input text.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <returns>Tokenized string array.</returns>
        private string[] GetTokens(string text)
        {
            if (string.IsNullOrEmpty(text) || _sortedTokens.Count == 0)
                return Array.Empty<string>();

            var result = new List<string>();
            int index = 0;
            int length = text.Length;

            while (index < length)
            {
                bool matched = false;
                // Match longest tokens first
                foreach (string token in _sortedTokens)
                {
                    int tokenLen = token.Length;
                    if (index + tokenLen <= length &&
                        string.Compare(text, index, token, 0, tokenLen, StringComparison.Ordinal) == 0)
                    {
                        result.Add(token);
                        index += tokenLen;
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    // Fallback to single character to avoid infinite loop
                    result.Add(text[index].ToString());
                    index++;
                }
            }
            return result.ToArray();
        }

        /// <summary>
        /// Builds token mapping dictionary and sorted token list from raw token lines.
        /// </summary>
        /// <param name="tokens">Raw token lines array.</param>
        /// <returns>Token dictionary and sorted list.</returns>
        private static (Dictionary<string, int>, List<string>) BuildTokenMappings(string[] tokens)
        {
            var dict = new Dictionary<string, int>(tokens.Length);
            var sorted = new List<string>(tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                string line = tokens[i];
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                // Split by tab/space to extract pure token text
                string token = line.Split(new char[] { '\t', ' ' })[0];
                if (!dict.ContainsKey(token))
                {
                    dict[token] = i;
                    sorted.Add(token);
                }
            }

            // Sort by length descending for maximum matching priority
            sorted.Sort((a, b) => b.Length.CompareTo(a.Length));
            return (dict, sorted);
        }
    }
}