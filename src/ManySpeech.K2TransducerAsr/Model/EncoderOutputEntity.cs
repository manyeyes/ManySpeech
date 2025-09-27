// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Collections;

namespace ManySpeech.K2TransducerAsr.Model
{
    public class EncoderOutputEntity
    {

        private float[]? _encoder_out;
        private long[]? _encoder_out_lens;
        private List<float[]>? _encoder_out_states;

        public float[]? encoder_out { get => _encoder_out; set => _encoder_out = value; }
        public long[]? encoder_out_lens { get => _encoder_out_lens; set => _encoder_out_lens = value; }
        public List<float[]>? encoder_out_states { get => _encoder_out_states; set => _encoder_out_states = value; }
    }
}
