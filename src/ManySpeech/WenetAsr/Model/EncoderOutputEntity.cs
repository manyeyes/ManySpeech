// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.WenetAsr.Model
{
    public class EncoderOutputEntity
    {
        private float[]? _output;
        private int _index;
        private float[]? _r_att_cache;
        private float[]? _r_cnn_cache;
        private List<float[]> statesList;

        public float[]? Output { get => _output; set => _output = value; }
        public float[]? R_att_cache { get => _r_att_cache; set => _r_att_cache = value; }
        public float[]? R_cnn_cache { get => _r_cnn_cache; set => _r_cnn_cache = value; }
        public List<float[]> StatesList { get => statesList; set => statesList = value; }
        public int Index { get => _index; set => _index = value; }
    }
}
