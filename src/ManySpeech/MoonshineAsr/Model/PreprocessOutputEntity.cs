// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.MoonshineAsr.Model
{
    public class PreprocessOutputEntity
    {
        private float[]? _seq;
        private int[]? _seqDim;

        public float[]? Seq { get => _seq; set => _seq = value; }
        public int[]? SeqDim { get => _seqDim; set => _seqDim = value; }
    }
}
