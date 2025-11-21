// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace ManySpeech.AudioSep.Model
{
    public class ModelOutputEntity
    { 
        private string _stemName=string.Empty;
        private float[]? _stemContents = null;

        public string StemName { get => _stemName; set => _stemName = value; }
        public float[]? StemContents { get => _stemContents; set => _stemContents = value; }
    }
}
