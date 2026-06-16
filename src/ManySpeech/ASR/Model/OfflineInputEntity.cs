// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.ASR.Model
{
    public class OfflineInputEntity
    {
        public float[]? Speech { get; set; }
        public int SpeechLength { get; set; }
        public List<string>? Hotwords { get; set; } = new List<string>();
        public string? Language { get; set; }
        public string? Region { get; set; }
    }
}
