using ManySpeech.K2TransducerAsr.Examples.Entities;

namespace ManySpeech.K2TransducerAsr.Examples.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}