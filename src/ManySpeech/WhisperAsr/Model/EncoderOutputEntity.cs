// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace ManySpeech.WhisperAsr.Model
{
    public class EncoderOutputEntity
    {
        private float[]? _output;
        private int[]? _dim;

        public float[]? Output { get => _output; set => _output = value; }
        public int[]? Dim { get => _dim; set => _dim = value; }
    }
}
