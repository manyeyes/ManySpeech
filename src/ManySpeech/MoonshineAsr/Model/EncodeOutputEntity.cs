// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.MoonshineAsr.Model
{
    public class EncodeOutputEntity
    {
        private float[]? _output;
        private int[] _outputDim;

        public float[]? Output { get => _output; set => _output = value; }
        public int[] OutputDim { get => _outputDim; set => _outputDim = value; }
    }
}
