// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.FireRedAsr.Model;
using SpeechFeatures;
using ManySpeech.FireRedAsr.Utils;

namespace ManySpeech.FireRedAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2025 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        private OnlineFbank _onlineFbank;
        private CmvnEntity _cmvnEntity;

        public WavFrontend(string mvnFilePath, FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                window_type: _frontendConfEntity.window
                );
            _cmvnEntity = LoadCmvn(mvnFilePath);
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            samples = samples.Select((float x) => x * 32768f).ToArray();
            float[] fbanks = _onlineFbank.GetFbank(samples);
            return fbanks;
        }
        public void InputFinished()
        {
            _onlineFbank.InputFinished();
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
                    inputs[dim * i + k] = (inputs[dim * i + k] - neg_mean[k]) * inv_stddev[k];
                }
            }
            return inputs;
        }
        private CmvnEntity LoadCmvn(string mvnFilePath)
        {
            List<double> means_list = new List<double>();
            List<double> vars_list = new List<double>();
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
                        means_list = add_shift_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => double.Parse(x.Trim())).ToList();
                        continue;
                    }
                    if (strLine.StartsWith("<LearnRateCoef>") && i == 2)
                    {
                        string[] rescale_line = strLine.Substring(strLine.IndexOf("[") + 1, strLine.LastIndexOf("]") - strLine.IndexOf("[") - 1).Split(' ');
                        vars_list = rescale_line.Where(x => !string.IsNullOrEmpty(x)).Select(x => double.Parse(x.Trim())).ToList();
                        continue;
                    }
                }
            }
            double count = means_list.Last();
            double floor = 1e-20;
#if NETSTANDARD2_0 || NET461_OR_GREATER
            means_list = means_list.Select(x => x / count).ToList().SkipLastOne().ToList();
#else
            means_list = means_list.Select(x => x / count).SkipLast(1).ToList();
#endif
            vars_list = vars_list.Zip(means_list, (a, b) => a / count - b * b).ToList();
            vars_list = vars_list.Select(x => (double)(x < floor ? floor : x)).ToList();
            vars_list = vars_list.Select(x => (double)(1.0F / Math.Sqrt(x))).ToList();
            CmvnEntity cmvnEntity = new CmvnEntity();
            cmvnEntity.Means = means_list;
            cmvnEntity.Vars = vars_list;
            return cmvnEntity;
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
