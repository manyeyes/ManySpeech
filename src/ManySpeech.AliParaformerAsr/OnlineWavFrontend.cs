// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using SpeechFeatures;

namespace ManySpeech.AliParaformerAsr
{
    /// <summary>
    /// OnlineWavFrontend
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class OnlineWavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        OnlineFbank _onlineFbank;
        private CmvnEntity _cmvnEntity;

        public OnlineWavFrontend(string mvnFilePath, FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                window_type: _frontendConfEntity.window,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels
                );
            _cmvnEntity = LoadCmvn(mvnFilePath);
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            float[] fbanks = _onlineFbank.GetFbank(samples);//GetFbankIndoor
            return fbanks;
        }


        public float[] LfrCmvn(float[] fbanks)
        {
            float[] features = fbanks;
            if (_frontendConfEntity.lfr_m != 1 || _frontendConfEntity.lfr_n != 1)
            {
                features = ApplyLfr(fbanks, _frontendConfEntity.lfr_m, _frontendConfEntity.lfr_n);
            }
            if (_cmvnEntity != null)
            {
                features = ApplyCmvn(features);
            }
            return features;
        }

        public float[] ApplyCmvn(float[] inputs)
        {
            var arr_neg_mean = _cmvnEntity.Means;
            float[] neg_mean = arr_neg_mean.Select(x => (float)Convert.ToDouble(x)).ToArray();
            var arr_inv_stddev = _cmvnEntity.Vars;
            float[] inv_stddev = arr_inv_stddev.Select(x => (float)Convert.ToDouble(x)).ToArray();

            int dim = neg_mean.Length;
            int num_frames = inputs.Length / dim;

            for (int i = 0; i < num_frames; i++)
            {
                for (int k = 0; k != dim; ++k)
                {
                    inputs[dim * i + k] = (inputs[dim * i + k] + neg_mean[k]) * inv_stddev[k];
                }
            }
            return inputs;
        }

        public float[] ApplyLfr(float[] inputs, int lfr_m, int lfr_n)
        {
            int t = inputs.Length / 80;
            int t_lfr = 0;
            if (t % lfr_n < lfr_m - lfr_n)
            {
                t_lfr = (int)Math.Floor((double)(t / lfr_n)) - 1;
            }
            if (t % lfr_n >= lfr_m - lfr_n)
            {
                t_lfr = (int)Math.Floor((double)(t / lfr_n));
            }
            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            for (int i = 0; i < t_lfr; i++)
            {
                Array.Copy(inputs, i * lfr_n * 80, LFR_outputs, i * lfr_m * 80, lfr_m * 80);
            }
            return LFR_outputs;
        }

        public float[] ApplyLfr2(float[] inputs, int lfr_m, int lfr_n)
        {
            int t = inputs.Length / 80;
            int t_lfr = (int)Math.Floor((double)(t / lfr_n));
            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            for (int i = 0; i < t_lfr; i++)
            {
                Array.Copy(inputs, i * lfr_n * 80, LFR_outputs, i * lfr_m * 80, lfr_m * 80);
            }
            return LFR_outputs;
        }

        private CmvnEntity LoadCmvn(string mvnFilePath)
        {
            List<float> means_list = new List<float>();
            List<float> vars_list = new List<float>();
            StreamReader srtReader = new StreamReader(mvnFilePath);
            int i = 0;
            while (!srtReader.EndOfStream)
            {
                string? strLine = srtReader.ReadLine();
                if (!string.IsNullOrEmpty(strLine))
                {
                    if (strLine.StartsWith("<AddShift>"))
                    {
                        i = 1;
                        continue;
                    }
                    if (strLine.StartsWith("<Rescale>"))
                    {
                        i = 2;
                        continue;
                    }
                    if (strLine.StartsWith("<LearnRateCoef>") && i == 1)
                    {
                        string[] add_shift_line = strLine.Substring(strLine.IndexOf("[") + 1, strLine.LastIndexOf("]") - strLine.IndexOf("[") - 1).Split(' ');
                        means_list = add_shift_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => float.Parse(x.Trim())).ToList();
                        //i++;
                        continue;
                    }
                    if (strLine.StartsWith("<LearnRateCoef>") && i == 2)
                    {
                        string[] rescale_line = strLine.Substring(strLine.IndexOf("[") + 1, strLine.LastIndexOf("]") - strLine.IndexOf("[") - 1).Split(' ');
                        vars_list = rescale_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => float.Parse(x.Trim())).ToList();
                        //i++;
                        continue;
                    }
                }
            }
            CmvnEntity cmvnEntity = new CmvnEntity();
            cmvnEntity.Means = means_list;
            cmvnEntity.Vars = vars_list;
            return cmvnEntity;
        }

        /// <summary>
        /// Streaming Positional encoding
        /// </summary>
        /// <returns></returns>
        public float[] SinusoidalPositionEncoder(float[] inputs, int timesteps, int inputsDim, int startIdx)
        {
            //forward
            float[] positions = new float[timesteps + startIdx];
            for (int i = 1; i < positions.Length + 1; i++)
            {
                positions[i - 1] = (float)i;
            }
            //forward
            //encode
            int batch_size = 1;
            float log_timescale_increment = (float)Math.Log(10000F) / (inputsDim / 2 - 1);
            float[] inv_timescales = new float[inputsDim / 2];
            for (int i = 0; i < inv_timescales.Length; i++)
            {
                inv_timescales[i] = (float)(i + 1);
            }
            inv_timescales = inv_timescales.Select(x => x * (-log_timescale_increment)).ToArray();
            inv_timescales = inv_timescales.Select(x => (float)Math.Exp(x)).ToArray();
            float[] scaled_time = new float[inv_timescales.Length * positions.Length * 2];
            foreach (float p in positions)
            {
                float[] scaled_time_item_sin = inv_timescales.Select(x => (float)Math.Sin(x * p)).ToArray();
                float[] scaled_time_item_cos = inv_timescales.Select(x => (float)Math.Cos(x * p)).ToArray();
                Array.Copy(scaled_time_item_sin, 0, scaled_time, ((int)p - 1) * (scaled_time_item_sin.Length + scaled_time_item_cos.Length), scaled_time_item_sin.Length);
                Array.Copy(scaled_time_item_cos, 0, scaled_time, ((int)p - 1) * (scaled_time_item_sin.Length + scaled_time_item_cos.Length) + scaled_time_item_sin.Length, scaled_time_item_cos.Length);
            }
            float[] encoding = scaled_time;
            float[] position_encoding = new float[inputs.Length];
            Array.Copy(encoding, inputsDim * startIdx, position_encoding, 0, position_encoding.Length);
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] += position_encoding[i];
            }
            return inputs;
            //encode
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_onlineFbank != null)
                {
                    _onlineFbank.Dispose();
                }
                if (_cmvnEntity != null)
                {
                    _cmvnEntity = null;
                }
                if (_frontendConfEntity != null)
                {
                    _frontendConfEntity = null;
                }
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
