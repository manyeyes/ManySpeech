using ManySpeech.AliFsmnVad.Model;
using SpeechFeatures;
using System.Reflection;

namespace ManySpeech.AliFsmnVad
{
    internal class WavFrontend : IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        OnlineFbank _onlineFbank;
        private CmvnEntity _cmvnEntity;

        public WavFrontend(string mvnFilePath, FrontendConfEntity frontendConfEntity)
        {
            //_mvnFilePath = mvnFilePath;
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: true,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels
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

        private float[] ApplyCmvn(float[] inputs)
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
            //List<float[]> LFR_inputs = new List<float[]>();
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
                    //float[] input_i = new float[lfr_m * 80];
                    //Array.Copy(inputs, i * lfr_n * 80, input_i, 0, input_i.Length);
                    //LFR_inputs.Add(input_i);
                    Array.Copy(inputs, i * lfr_n * 80, LFR_outputs, i * lfr_m * 80, lfr_m * 80);
                }
                else
                {
                    // process last LFR frame
                    int num_padding = lfr_m - (t - i * lfr_n);
                    float[] frame = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, frame, 0, (t - i * lfr_n) * 80);//frame.Length

                    for (int j = 0; j < num_padding; j++)
                    {
                        Array.Copy(inputs, (t - 1) * 80, frame, (lfr_m - num_padding + j) * 80, 80);
                    }
                    //LFR_inputs.Add(frame);
                    Array.Copy(frame, 0, LFR_outputs, i * lfr_m * 80, frame.Length);
                }
            }
            return LFR_outputs;
        }

        public float[] ApplyLfr_originalVersion(float[] inputs, int lfr_m, int lfr_n)
        {
            List<float[]> LFR_inputs = new List<float[]>();

            int t = inputs.Length / 80;
            int t_lfr = (int)Math.Floor((double)(t / lfr_n));
            float[] input_0 = new float[80];
            Array.Copy(inputs, 0, input_0, 0, 80);
            int tile_x = (lfr_m - 1) / 2;
            //for(int i = 0; i < tile_x; i++)
            //{
            //    LFR_inputs.Add(input_0);
            //}
            //float[] left_padding = np.tile(inputs[0], ((lfr_m - 1) / 2, 1));
            //float[] inputs = np.vstack((left_padding, inputs));
            t = t + tile_x;
            float[] inputs_temp = new float[t * 80];
            for (int i = 0; i < tile_x; i++)
            {
                Array.Copy(input_0, 0, inputs_temp, tile_x * 80, 80);
            }
            //Array.Copy(input_0, 0, inputs_temp, 0, 80);
            //Array.Copy(input_0, 0, inputs_temp, 80, 80);
            Array.Copy(inputs, 0, inputs_temp, tile_x * 80, inputs.Length);
            inputs = inputs_temp;
            for (int i = 0; i < t_lfr; i++)
            {
                if (lfr_m <= t - i * lfr_n)
                {
                    float[] input_i = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, input_i, 0, input_i.Length);
                    LFR_inputs.Add(input_i);
                }
                else
                {
                    // process last LFR frame
                    int num_padding = lfr_m - (t - i * lfr_n);
                    float[] frame = new float[lfr_m * 80];
                    Array.Copy(inputs, i * lfr_n * 80, frame, 0, (t - i * lfr_n) * 80);//frame.Length

                    for (int j = 0; j < num_padding; j++)
                    {
                        Array.Copy(inputs, (t - 1) * 80, frame, (lfr_m - num_padding + j) * 80, 80);
                    }
                    LFR_inputs.Add(frame);
                }
            }
            inputs = null;
            float[] LFR_outputs = new float[t_lfr * lfr_m * 80];
            int p = 0;
            foreach (float[] f in LFR_inputs)
            {
                Array.Copy(f, 0, LFR_outputs, p * lfr_m * 80, f.Length);
                p++;
            }
            return LFR_outputs;
        }

        private CmvnEntity LoadCmvn(string mvnFilePath)
        {           
            List<float> means_list = new List<float>();
            List<float> vars_list = new List<float>();
            StreamReader? streamReader = null;
            if (!string.IsNullOrEmpty(mvnFilePath) && mvnFilePath.IndexOf("/") < 0 && mvnFilePath.IndexOf("\\") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(mvnFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{mvnFilePath}' not found.");
                streamReader = new StreamReader(stream);
            }
            else if (File.Exists(mvnFilePath))
            {
                streamReader = new StreamReader(mvnFilePath);
            }
            if (streamReader == null) {
                return null;
            }
            int i = 0;
            while (!streamReader.EndOfStream)
            {
                string? strLine = streamReader.ReadLine();
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
            streamReader.Close();
            CmvnEntity cmvnEntity = new CmvnEntity();
            cmvnEntity.Means = means_list;
            cmvnEntity.Vars = vars_list;
            return cmvnEntity;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onlineFbank != null)
                    {
                        _onlineFbank.Dispose();
                    }
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~WavFrontend()
        {
            Dispose(_disposed);
        }
    }
}
