using ManySpeech.AliParaformerAsr.Examples.Entities;

namespace ManySpeech.AliParaformerAsr.Examples.Delegates
{
    public delegate void RecognitionResultCallback(AsrResultEntity result);
    public delegate void RecognitionCompletedCallback(TimeSpan totalTime, TimeSpan totalDuration, int processedCount, float[]? sample);
}