namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            test_OfflineLanguageID();
            test_OfflineRecognizer();
            test_TranscribeRecognizer();
            test_OnlineRecognizer();
        }
    }
}