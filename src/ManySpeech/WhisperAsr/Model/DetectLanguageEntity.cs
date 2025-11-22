// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace ManySpeech.WhisperAsr.Model
{
    public class DetectLanguageEntity
    {
        private List<string> _languageCodes = new List<string>();
        public List<string> LanguageCodes { get => _languageCodes; set => _languageCodes = value; }
    }
}
