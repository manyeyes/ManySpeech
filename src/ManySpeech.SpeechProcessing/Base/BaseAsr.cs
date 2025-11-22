using ManySpeech.SpeechProcessing.Delegates;
using ManySpeech.SpeechProcessing.Entities;
using ManySpeech.SpeechProcessing.ASR.Base;

namespace ManySpeech.SpeechProcessing.Base
{
    public abstract class BaseAsr:IDisposable,IRecognizer
    {
        public bool _disposed=false;

        public string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        public event RecognitionResultCallback OnRecognitionResult;
        public event RecognitionCompletedCallback OnRecognitionCompleted;
        protected void RaiseRecognitionResult(AsrResultEntity result)
        {
            OnRecognitionResult?.Invoke(result);
        }
        protected void RaiseRecognitionCompleted(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample = null)
        {
            OnRecognitionCompleted?.Invoke(totalTime, totalDuration, processedCount, sample);
        }
        public void ResetRecognitionResultHandlers()
        {
            OnRecognitionResult = null;
        }
        public void ResetRecognitionCompletedHandlers()
        {
            OnRecognitionCompleted = null;
        }
        public abstract Task<List<AsrResultEntity>> RecognizeAsync(
            List<List<float[]>> samplesList,
            string modelBasePath,
            string modelName = "",
            string modelAccuracy = "int8",
            string streamDecodeMethod = "one",
            int threadsNum = 2);
        protected virtual double CalculateAudioDuration(float[] samples, int sampleRate = 16000)
        {
            return samples.Length / (double)sampleRate * 1000; // 返回毫秒
        }
        public abstract void Dispose();
    }
}
