namespace ManySpeech.SpeechLid.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            OfflineFireRedAsrLanguage.FireRedAsrLanguageDetection();
            //OfflineWhisperAsrLanguage.WhisperAsrLanguageDetection();
        }
    }
}