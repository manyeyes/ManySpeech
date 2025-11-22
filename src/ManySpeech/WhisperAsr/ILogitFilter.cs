using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    internal interface ILogitFilter
    {
        internal void apply(ref List<List<float>> logits, List<List<Int64>> tokens);
    }
    internal class SuppressBlank : ILogitFilter
    {
        private Tokenizer _tokenizer;
        private int _sample_begin;
        internal SuppressBlank(Tokenizer tokenizer, int sample_begin)
        {
            _tokenizer = tokenizer;
            _sample_begin = sample_begin;
        }
        public void apply(ref List<List<float>> logits, List<List<Int64>> tokens)
        {
            if (tokens[0].Count == _sample_begin)
            {
                List<int> xxx = _tokenizer.encode(" ");
                xxx.Add(_tokenizer.Eot);
                foreach (List<float> floats in logits)
                {
                    foreach (int i in xxx)
                    {
                        floats[i] = float.NegativeInfinity;
                    }
                }
            }
        }
    }
    internal class SuppressTokens : ILogitFilter
    {
        int[] _suppress_tokens = null;
        internal SuppressTokens(int[] suppress_tokens)
        {
            _suppress_tokens = suppress_tokens;
        }
        public void apply(ref List<List<float>> logits, List<List<Int64>> tokens)
        {
            foreach (List<float> floats in logits)
            {
                foreach (int i in _suppress_tokens)
                {
                    floats[i] = float.NegativeInfinity;
                }
            }
        }
    }
    internal class ApplyTimestampRules : ILogitFilter
    {
        private Tokenizer _tokenizer;
        private int _sample_begin;
        private int? _max_initial_timestamp_index = null;
        internal ApplyTimestampRules(Tokenizer tokenizer, int sample_begin, int? max_initial_timestamp_index)
        {
            _tokenizer = tokenizer;
            _sample_begin = sample_begin;
            _max_initial_timestamp_index = max_initial_timestamp_index;
        }
        public void apply(ref List<List<float>> logits, List<List<Int64>> tokens)
        {
            // suppress <|notimestamps|> which is handled by without_timestamps
            if (_tokenizer.NoTimestamps != null && _tokenizer.NoTimestamps != int.MinValue)
            {
                foreach (List<float> floats in logits)
                {
                    floats[_tokenizer.NoTimestamps] = float.NegativeInfinity;
                }
            }
            for (int k = 0; k < tokens.Count; k++)
            {
                List<Int64> sampled_tokens = new List<Int64>(tokens[k]);
                List<Int64> sampled_tokens_temp = new List<Int64>();
                for (int i = _sample_begin; i < sampled_tokens.Count; i++)
                {
                    //sampled_tokens.RemoveAt(i);
                    sampled_tokens_temp.Add(sampled_tokens[i]);
                }
                sampled_tokens = new List<long>(sampled_tokens_temp);
                sampled_tokens_temp.Clear();
                ////////////////////////////
                Int64[] seq = sampled_tokens.ToArray();
                bool last_was_timestamp = seq.Length >= 1 && seq.Last() >= _tokenizer.TimestampBegin;
                bool penultimate_was_timestamp = seq.Length < 2 || seq[seq.Length - 2] >= _tokenizer.TimestampBegin;
                if (last_was_timestamp)
                {
                    if (penultimate_was_timestamp)
                    {
                        for (int i = 0; i < logits[k].Count; i++)
                        {
                            if (i >= _tokenizer.TimestampBegin)
                            {
                                logits[k][i] = float.NegativeInfinity;
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < logits[k].Count; i++)
                        {
                            if (i < _tokenizer.Eot)
                            {
                                logits[k][i] = float.NegativeInfinity;
                            }
                        }
                    }
                }
                List<Int64> timestamps = sampled_tokens.Where(x => x >= _tokenizer.TimestampBegin).ToList();
                if (timestamps.Count > 0)
                {
                    Int64 timestamp_last = 0;
                    // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                    // also force each segment to have a nonzero length, to prevent infinite looping
                    if (last_was_timestamp && !penultimate_was_timestamp)
                    {
                        timestamp_last = timestamps.Last();
                    }
                    else
                    {
                        timestamp_last = timestamps.Last() + 1;
                    }
                    //logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf;
                    for (int i = 0; i < logits[k].Count; i++)
                    {
                        if (i > _tokenizer.TimestampBegin && i < timestamp_last)
                        {
                            logits[k][i] = float.NegativeInfinity; 
                        }
                    }
                }
            }
            if (tokens[0].Count == _sample_begin)
            {
                for (int k = 0; k < logits.Count; k++)
                {
                    for (int i = 0; i < logits[k].Count; i++)
                    {
                        if (i < _tokenizer.TimestampBegin)
                        {
                            logits[k][i] = float.NegativeInfinity;
                        }
                    }
                }
                if (_max_initial_timestamp_index != null)
                {
                    int? last_allowed = _tokenizer.TimestampBegin + _max_initial_timestamp_index;
                    for (int k = 0; k < logits.Count; k++)
                    {
                        for (int i = 0; i < logits[k].Count; i++)
                        {
                            if (i >= last_allowed+1)
                            {
                                logits[k][i] = float.NegativeInfinity;
                            }
                        }
                    }
                }
            }
            // if sum of probability over timestamps is above any other token, sample timestamp
            List<List<float>> logprobs = logits.Select(x => x = ComputeHelper.LogCompute(ComputeHelper.SoftmaxCompute(x.ToArray())).ToList()).ToList();
            for (int k = 0; k < logprobs.Count; k++)
            {
                List<float> logprobs_timestamp_logprob = new List<float>();
                for (int i = 0; i < logprobs[k].Count; i++)
                {
                    if (i >= _tokenizer.TimestampBegin)
                    {
                        logprobs_timestamp_logprob.Add(logprobs[k][i]);
                    }
                }
                float timestamp_logprob = ComputeHelper.LogSumExp(logprobs_timestamp_logprob.ToArray());
                //max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
                List<float> logprobs_max_text_token_logprob = new List<float>();
                for (int i = 0; i < logprobs[k].Count; i++)
                {
                    if (i < _tokenizer.TimestampBegin)
                    {
                        logprobs_max_text_token_logprob.Add(logprobs[k][i]);
                    }
                }
                float max_text_token_logprob = logprobs_max_text_token_logprob.Max();
                if (timestamp_logprob > max_text_token_logprob)
                {
                    //logits[k, : self.tokenizer.timestamp_begin] = -np.inf
                    for (int i = 0; i < logits[k].Count; i++)
                    {
                        if (i < _tokenizer.TimestampBegin)
                        {
                            logits[k][i] = float.NegativeInfinity;
                        }
                    }
                }
            }
        }
    }

}
