using ManySpeech.K2TransducerAsr.Examples.Delegates;
using ManySpeech.K2TransducerAsr.Examples.Entities;

namespace ManySpeech.K2TransducerAsr.Examples.Base
{
    internal class BaseAsr
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        public static event RecognitionResultCallback OnRecognitionResult;
        public static event RecognitionCompletedCallback OnRecognitionCompleted;
        protected static void RaiseRecognitionResult(AsrResultEntity result)
        {
            OnRecognitionResult?.Invoke(result);
        }
        protected static void RaiseRecognitionCompleted(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample = null)
        {
            OnRecognitionCompleted?.Invoke(totalTime, totalDuration, processedCount, sample);
        }
        public static void ResetRecognitionResultHandlers()
        {
            OnRecognitionResult = null;
        }
        public static void ResetRecognitionCompletedHandlers()
        {
            OnRecognitionCompleted = null;
        }
    }
}
