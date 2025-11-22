// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

namespace ManySpeech.K2TransducerAsr.Model
{
    public class DecoderOutputEntity
    {
        private float[]? _decoder_out;
        public float[]? decoder_out { get => _decoder_out; set => _decoder_out = value; }
    }
}
