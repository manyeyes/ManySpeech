namespace ManySpeech.MoonshineAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            test_MoonshineAsrOfflineRecognizer();
            test_MoonshineAsrOnlineRecognizer();
            test_MoonshineAsrOnlineVadRecognizer();
        }
    }
}