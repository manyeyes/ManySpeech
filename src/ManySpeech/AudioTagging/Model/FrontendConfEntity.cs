// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
namespace ManySpeech.AudioTagging.Model
{
    public class FrontendConfEntity
    {
        private int _fs = 16000;
        private string _window = "hanning";
        private int _n_mels = 64;
        private int _frame_length = 32;
        private int _frame_shift = 10;
        private float _dither = 0F;
        private int _lfr_m = 7;
        private int _lfr_n = 6;
        private bool _snip_edges = false;
        private bool _is_librosa = false;
        private bool _htk_mode = false;
        private float _low_freq = 0F;
        private float _high_freq = 0F;
        private string _norm = "";
        private bool _remove_dc_offset = false;
        private float _preemph_coeff = 0f;
        private bool _use_log_fbank=false;

        public int fs { get => _fs; set => _fs = value; }
        public string window { get => _window; set => _window = value; }
        public int n_mels { get => _n_mels; set => _n_mels = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_shift { get => _frame_shift; set => _frame_shift = value; }
        public float dither { get => _dither; set => _dither = value; }
        public int lfr_m { get => _lfr_m; set => _lfr_m = value; }
        public int lfr_n { get => _lfr_n; set => _lfr_n = value; }
        public bool snip_edges { get => _snip_edges; set => _snip_edges = value; }
        public bool is_librosa { get => _is_librosa; set => _is_librosa = value; }
        public bool htk_mode { get => _htk_mode; set => _htk_mode = value; }
        public float low_freq { get => _low_freq; set => _low_freq = value; }
        public float high_freq { get => _high_freq; set => _high_freq = value; }
        public string norm { get => _norm; set => _norm = value; }
        public bool remove_dc_offset { get => _remove_dc_offset; set => _remove_dc_offset = value; }
        public float preemph_coeff { get => _preemph_coeff; set => _preemph_coeff = value; }
        public bool use_log_fbank { get => _use_log_fbank; set => _use_log_fbank = value; }
    }
}
