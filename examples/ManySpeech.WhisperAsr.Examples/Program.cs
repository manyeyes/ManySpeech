namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        [STAThread]
        private static void Main()
        {
            OfflineWhisperAsrLanguageID.OfflineLanguageID();
            OfflinneWhisperAsrRecognizer.OfflineRecognizer();
            OnlineWhisperAsrTranscriber.TranscribeRecognizer();
            OnlineWhisperAsrRecognizer.OnlineRecognizer();
        }
    }
}