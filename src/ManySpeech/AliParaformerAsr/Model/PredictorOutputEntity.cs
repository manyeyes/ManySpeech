// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Collections;

namespace ManySpeech.AliParaformerAsr.Model
{
    public class PredictorOutputEntity
    {

        private float[] _acoustic_embeds;
        private int[] _acoustic_embeds_len;

        public float[] Acoustic_embeds { get => _acoustic_embeds; set => _acoustic_embeds = value; }
        public int[] Acoustic_embeds_len { get => _acoustic_embeds_len; set => _acoustic_embeds_len = value; }
    }
}
