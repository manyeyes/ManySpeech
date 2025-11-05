using ManySpeech.SpeechProcessing.Entities;

namespace ManySpeech.SpeechProcessing.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}