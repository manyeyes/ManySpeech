namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            test_WhisperAsrLanguageDetection();
            test_WhisperAsrOfflineRecognizer();
            test_WhisperAsrTranscribeRecognizer();
            test_WhisperAsrOnlineRecognizer();
        }
    }
}