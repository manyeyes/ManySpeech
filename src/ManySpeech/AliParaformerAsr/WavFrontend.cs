// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using SpeechFeatures;

namespace ManySpeech.AliParaformerAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConf? _frontendConf;
        private OnlineFbank? _onlineFbank;
        private CmvnEntity? _cmvnEntity;

        //public WavFrontend(string mvnFilePath, FrontendConf frontendConfEntity)
        public WavFrontend(FrontendConf? frontendConf = null, string? mvnFilePath=null)
        {
            if (frontendConf != null)
            {
                _frontendConf = frontendConf;
                _onlineFbank = new OnlineFbank(
                    dither: _frontendConf.dither,
                    snip_edges: _frontendConf.snip_edges,
                    window_type: _frontendConf.window,
                    sample_rate: _frontendConf.fs,
                    num_bins: _frontendConf.n_mels
                    );
            }
            if (string.IsNullOrEmpty(mvnFilePath))
            {
                _cmvnEntity = LoadCmvn(mvnFilePath);
            }
        }

        public float[] GetFeatures(float[] samples)
        {
            float[] features = samples.Select((float x) => x * 32768f).ToArray();
            if (_onlineFbank != null)
            {
                features = _onlineFbank.GetFbank(features);
            }
            return features;
        }

        public float[] LfrCmvn(float[] fbanks)
        {
            float[] features = fbanks;
            if (_frontendConf != null)
            {
                if (_frontendConf.lfr_m != 1 || _frontendConf.lfr_n != 1)
                {
                    features = ApplyLfr(fbanks, _frontendConf.lfr_m, _frontendConf.lfr_n);
                }
            }
            features = ApplyCmvn(features);
            return features;
        }

        public float[] ApplyCmvn(float[] inputs)
        {
            if (_cmvnEntity != null)
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
            }
            return inputs;
        }

        public float[] ApplyLfr(float[] inputs, int lfr_m, int lfr_n)
        {
            int t = inputs.Length / 80;
            int t_lfr = (int)Math.Floor((double)(t / lfr_n));
            float[] input_0 = new float[80];
            Array.Copy(inputs, 0, input_0, 0, 80);
            int tile_x = (lfr_m - 1) / 2;
            t = t + tile_x;
            float[] inputs_temp = new float[t * 80];
            for (int i = 0; i < tile_x; i++)
            {
                Array.Copy(input_0, 0, inputs_temp, tile_x * 80, 80);
            }
            Array.Copy(inputs, 0, inputs_temp, tile_x * 80, inputs.Length);
            inputs = inputs_temp;

            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            for (int i = 0; i < t_lfr; i++)
            {
                if (lfr_m <= t - i * lfr_n)
                {
                    Array.Copy(inputs, i * lfr_n * 80, LFR_outputs, i * lfr_m * 80, lfr_m * 80);
                }
                else
                {
                    // process last LFR frame
                    int num_padding = lfr_m - (t - i * lfr_n);
                    float[] frame = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, frame, 0, (t - i * lfr_n) * 80);

                    for (int j = 0; j < num_padding; j++)
                    {
                        Array.Copy(inputs, (t - 1) * 80, frame, (lfr_m - num_padding + j) * 80, 80);
                    }
                    Array.Copy(frame, 0, LFR_outputs, i * lfr_m * 80, frame.Length);
                }
            }
            return LFR_outputs;
        }
        private CmvnEntity? LoadCmvn(string? mvnFilePath)
        {
            if(string.IsNullOrEmpty(mvnFilePath) || !File.Exists(mvnFilePath))
            {
                return null;
            }
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
        public void InputFinished()
        {
            if (_onlineFbank != null)
            {
                _onlineFbank.InputFinished();
            }
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_onlineFbank != null)
                {
                    _onlineFbank.Dispose();
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
