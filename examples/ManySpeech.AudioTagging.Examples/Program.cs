namespace ManySpeech.AudioTagging.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            OfflineAudioTagging.OfflineOfflineTagging();
        }
    }
}