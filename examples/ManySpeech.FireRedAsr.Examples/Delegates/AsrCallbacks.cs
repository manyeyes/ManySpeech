using ManySpeech.FireRedAsr.Examples.Entities;

namespace ManySpeech.FireRedAsr.Examples.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}