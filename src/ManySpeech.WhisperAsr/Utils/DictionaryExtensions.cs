namespace ManySpeech.WhisperAsr.Utils
{
    internal static class DictionaryExtensions
    {
        // Overload 1: When no default value is specified, returns the default value of TValue (e.g. null)
        public static TValue GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key)
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            dictionary.TryGetValue(key, out var value);
            return value;
        }

        // Overload 2: Allows specifying a custom default value
        public static TValue GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key,
            TValue defaultValue)
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
        }
    }
}
