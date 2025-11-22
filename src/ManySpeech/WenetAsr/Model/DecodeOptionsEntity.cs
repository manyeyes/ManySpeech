namespace ManySpeech.WenetAsr.Model
{
    // LatticeFasterDecoderConfig has the following key members
    // beam: decoding beam
    // max_active: Decoder max active states
    // lattice_beam: Lattice generation beam
    internal class CtcWfstBeamSearchOptions
    {
        private float acoustic_scale = 1.0f;
        private float nbest = 10;
        // When blank score is greater than this thresh, skip the frame in viterbi
        // search
        private float blank_skip_thresh = 0.98f;
        private float blank_scale = 1.0f;

        public float Acoustic_scale { get => acoustic_scale; set => acoustic_scale = value; }
        public float Nbest { get => nbest; set => nbest = value; }
        public float Blank_skip_thresh { get => blank_skip_thresh; set => blank_skip_thresh = value; }
        public float Blank_scale { get => blank_scale; set => blank_scale = value; }
    }
    internal class CtcPrefixBeamSearchOptions
    {
        private int _blank = 0;  // blank id
        private int _first_beam_size = 10;
        private int _second_beam_size = 10;

        public int Blank { get => _blank; set => _blank = value; }
        public int First_beam_size { get => _first_beam_size; set => _first_beam_size = value; }
        public int Second_beam_size { get => _second_beam_size; set => _second_beam_size = value; }
    }
    internal class CtcEndpointRule
    {
        private bool _must_decoded_sth;
        private int _min_trailing_silence;
        private int _min_utterance_length;

        public CtcEndpointRule(bool must_decoded_sth = true, int min_trailing_silence = 1000,
                        int min_utterance_length = 0)
        {
            _must_decoded_sth = must_decoded_sth;
            _min_trailing_silence = min_trailing_silence;
            _min_utterance_length = min_utterance_length;
        }

        public bool Must_decoded_sth { get => _must_decoded_sth; set => _must_decoded_sth = value; }
        public int Min_trailing_silence { get => _min_trailing_silence; set => _min_trailing_silence = value; }
        public int Min_utterance_length { get => _min_utterance_length; set => _min_utterance_length = value; }
    };
    internal class CtcEndpointConfig
    {
        /// We consider blank as silence for purposes of endpointing.
        private int _blank = 0;                // blank id
        private float _blank_threshold = 0.8f;  // blank threshold to be silence
        /// We support three rules. We terminate decoding if ANY of these rules
        /// evaluates to "true". If you want to add more rules, do it by changing this
        /// code. If you want to disable a rule, you can set the silence-timeout for
        /// that rule to a very large number.

        /// rule1 times out after 5000 ms of silence, even if we decoded nothing.
        private CtcEndpointRule _rule1;
        /// rule2 times out after 1000 ms of silence after decoding something.
        private CtcEndpointRule _rule2;
        /// rule3 times out after the utterance is 20000 ms long, regardless of
        /// anything else.
        private CtcEndpointRule _rule3;

        public CtcEndpointConfig()
        {
            _rule1 = new CtcEndpointRule(false, 5000, 0);
            _rule2 = new CtcEndpointRule(true, 1000, 0);
            _rule3 = new CtcEndpointRule(false, 0, 20000);
        }

        public int Blank { get => _blank; set => _blank = value; }
        public float Blank_threshold { get => _blank_threshold; set => _blank_threshold = value; }
        internal CtcEndpointRule Rule1 { get => _rule1; set => _rule1 = value; }
        internal CtcEndpointRule Rule2 { get => _rule2; set => _rule2 = value; }
        internal CtcEndpointRule Rule3 { get => _rule3; set => _rule3 = value; }
    }
    internal class DecodeOptionsEntity
    {
        // chunk_size is the frame number of one chunk after subsampling.
        // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
        // one chunk are 64 = 16*4
        private int _chunk_size = 16;
        private int _num_left_chunks = -1;

        // final_score = rescoring_weight * rescoring_score + ctc_weight * ctc_score;
        // rescoring_score = left_to_right_score * (1 - reverse_weight) +
        // right_to_left_score * reverse_weight
        // Please note the concept of ctc_scores in the following two search
        // methods are different.
        // For CtcPrefixBeamSearch, it's a sum(prefix) score + context score
        // For CtcWfstBeamSearch, it's a max(viterbi) path score + context score
        // So we should carefully set ctc_weight according to the search methods.
        private float _ctc_weight = 0.5f;
        private float _rescoring_weight = 1.0f;
        private float _reverse_weight = 0.0f;
        private CtcEndpointConfig _ctc_endpoint_config;
        private CtcPrefixBeamSearchOptions _ctc_prefix_search_opts;
        private CtcWfstBeamSearchOptions _ctc_wfst_search_opts;

        public int Chunk_size { get => _chunk_size; set => _chunk_size = value; }
        public int Num_left_chunks { get => _num_left_chunks; set => _num_left_chunks = value; }
        public float Ctc_weight { get => _ctc_weight; set => _ctc_weight = value; }
        public float Rescoring_weight { get => _rescoring_weight; set => _rescoring_weight = value; }
        public float Reverse_weight { get => _reverse_weight; set => _reverse_weight = value; }
        internal CtcEndpointConfig Ctc_endpoint_config { get => _ctc_endpoint_config; set => _ctc_endpoint_config = value; }
        internal CtcPrefixBeamSearchOptions Ctc_prefix_search_opts { get => _ctc_prefix_search_opts; set => _ctc_prefix_search_opts = value; }
        internal CtcWfstBeamSearchOptions Ctc_wfst_search_opts { get => _ctc_wfst_search_opts; set => _ctc_wfst_search_opts = value; }
    }
}
