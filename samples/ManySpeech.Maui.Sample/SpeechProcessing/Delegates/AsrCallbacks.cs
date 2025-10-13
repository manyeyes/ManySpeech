using ManySpeech.Maui.Sample.SpeechProcessing.Entities;

namespace ManySpeech.Maui.Sample.SpeechProcessing.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}