using ManySpeech.WenetAsr.Examples.Entities;

namespace ManySpeech.WenetAsr.Examples.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}