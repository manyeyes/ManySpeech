using ManySpeech.SpeechProcessing.Entities;
using ManySpeech.SpeechProcessing.Delegates;

namespace ManySpeech.SpeechProcessing.ASR.Base
{
    // 定义所有识别器的通用接口
    public interface IRecognizer
    {
        event RecognitionResultCallback OnRecognitionResult;
        event RecognitionCompletedCallback OnRecognitionCompleted;
        void ResetRecognitionResultHandlers();
        void ResetRecognitionCompletedHandlers();
        Task<List<AsrResultEntity>> RecognizeAsync(
            List<List<float[]>> samplesList,
            string modelBasePath,
            string modelName = "",
            string modelAccuracy = "int8",
            string streamDecodeMethod = "one",
            int threadsNum = 2);
        void Dispose();
    }

}
