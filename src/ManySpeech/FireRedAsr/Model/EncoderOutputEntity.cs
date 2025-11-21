// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
namespace ManySpeech.FireRedAsr.Model
{
    public class EncoderOutputEntity
    {
        private float[]? _output;
        private Int64[] _outputLengths;
        private bool[]? _mask;

        public float[]? Output { get => _output; set => _output = value; }
        public long[] OutputLengths { get => _outputLengths; set => _outputLengths = value; }
        public bool[]? Mask { get => _mask; set => _mask = value; }
    }
}
