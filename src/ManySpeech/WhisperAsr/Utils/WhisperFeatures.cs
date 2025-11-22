using System.Diagnostics;

namespace ManySpeech.WhisperAsr.Utils
{
    class WhisperMel
    {
        Int64 _n_len;
        Int64 _n_len_org;
        int _n_mel;

        float[]? _data = null;

        public Int64 N_len { get => _n_len; set => _n_len = value; }
        public Int64 N_len_org { get => _n_len_org; set => _n_len_org = value; }
        public int N_mel { get => _n_mel; set => _n_mel = value; }
        public float[]? Data { get => _data; set => _data = value; }
    };

    class WhisperMelFilters
    {
        int _n_mel;
        int _n_fft;
        float[]? _data = null;

        public int N_mel { get => _n_mel; set => _n_mel = value; }
        public int N_fft { get => _n_fft; set => _n_fft = value; }
        public float[]? Data { get => _data; set => _data = value; }
    }

    internal class WhisperFeatures : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;
        public static string _applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        // hard-coded audio hyperparameters
        private static int _sampleRate = 16000;
        private static int _nFFT = 400;
        private static int _hopLength = 160;
        private static int _chunkLength = 30;// 30-second 
        private static int nSamples = _chunkLength * _sampleRate;  // 480000 samples in a 30-second chunk
        private static int _nFrames = ComputeHelper.ExactDiv(nSamples, _hopLength);  // 3000 frames in a mel spectrogram input

        private static int _nSamplesPerToken = _hopLength * 2;  // the initial convolutions has stride 2
        private static int _framesPerSecond = ComputeHelper.ExactDiv(_sampleRate, _hopLength);  // 10ms per audio frame
        private static int _tokensPerSecond = ComputeHelper.ExactDiv(_sampleRate, _nSamplesPerToken);  // 20ms per audio token

        private static int _sinCosNCount = _nFFT;

        private float[] _sinVals = new float[_sinCosNCount];
        private float[] _cosVals = new float[_sinCosNCount];
        private WhisperMelFilters _whisperMelFilters = new WhisperMelFilters();
        private static int _nMels = 80;//nMels in {80, 128}
        private int _threadsNum = 1;

        public static int FramesPerSecond { get => _framesPerSecond; }
        public static int NFrames { get => _nFrames; }
        public static int HopLength { get => _hopLength; }
        public static int SampleRate { get => _sampleRate; }
        public static int NMels { get => _nMels; }

        public WhisperFeatures(int nMels = 80, int threadsNum = 1, string? melFiltersFilePath = null)
        {
            Debug.Assert(new int[] { 80, 128 }.Contains(nMels), string.Format("Unsupported n_mels: {0}", nMels));
            _nMels = nMels;
            _threadsNum = threadsNum;
            InitState();
            InitMelFilters(melFiltersFilePath);
        }

        public float[] LogMelSpectrogram(float[] audio, int padding = 0)
        {
            int n_mel = _nMels;
            int n_threads = _threadsNum;
            int frame_step = _hopLength;
            int frame_size = _nFFT;
            // Calculate the length of padding
            Int64 stage_1_pad = _sampleRate * 30;
            Int64 stage_2_pad = frame_size / 2;
            int n_samples = audio.Length;
            // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
            float[] samples_padded = new float[n_samples + stage_1_pad + stage_2_pad * 2];
            
            Array.Copy(audio, 0, samples_padded, stage_2_pad, n_samples);
            Array.Copy(audio, 0, samples_padded, 0, stage_2_pad);
            // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
            // limit 3000 frames
            float[] newAudio = new float[stage_1_pad + stage_2_pad * 2];
            Array.Copy(samples_padded, 0, newAudio, 0, newAudio.Length);
            samples_padded = newAudio;
            //if (samples_padded.Length > 0)
            //{   //F.pad(samples_padded, (0, padding)) 每个维度（行），右边填充padding个0
            //    float[] newAudio = new float[samples_padded.Length + padding];
            //    Array.Copy(samples_padded, 0, newAudio, 0, samples_padded.Length);
            //    samples_padded = newAudio;
            //}
            // Calculate number of frames + remove the last frame
            //Int64 n_len = (samples_padded.Length - stage_1_pad - frame_size) / frame_step;
            Int64 n_len = (samples_padded.Length - frame_size) / frame_step;

            WhisperMel whisperMel = new WhisperMel();
            whisperMel.N_len = n_len;
            whisperMel.N_mel = n_mel;
            whisperMel.Data = new float[n_mel * n_len];//new float[1100 + 79 * 1100];

            var hann = HannWindow(length: frame_size, periodic: true);
            Task[] tasks = new Task[n_threads];
            for (int iw = 0; iw < n_threads; iw++)
            {
                int ith = iw;
                Task task = new Task(() => LogMelSpectrogramThreads(ith, samples_padded, hann, ref whisperMel, frame_size: frame_size, frame_step: frame_step, threadsNum: n_threads));
                task.Start();
                tasks[iw] = task;
            }
            Task.WaitAll(tasks);

            // clamping and normalization
            double mmax = -1e20;
            for (int m = 0; m < n_mel * n_len; m++)
            {
                if (whisperMel.Data[m] > mmax)
                {
                    mmax = whisperMel.Data[m];
                }
            }

            mmax -= 8.0d;

            for (int m = 0; m < n_mel * n_len; m++)
            {
                if (whisperMel.Data[m] < mmax)
                {
                    whisperMel.Data[m] = (float)mmax;
                }

                whisperMel.Data[m] = (float)((whisperMel.Data[m] + 4.0) / 4.0);
            }
            return whisperMel.Data;
        }
        public void LogMelSpectrogramThreads(int ith, float[] audio, float[] hann, ref WhisperMel whisperMel, int frame_size, int frame_step, int threadsNum = 1, int padding = 0)
        {
            int n_samples = audio.Length;

            Int64 n_len = whisperMel.N_len;
            Int64 n_mel = whisperMel.N_mel;

            float[] fft_in = new float[frame_size];
            float[] fft_out = new float[frame_step * 2];

            int n_fft = 1 + (frame_size / 2);
            int i = ith;// ith;

            // calculate FFT only when fft_in are not all zero
            for (; i < Math.Min(n_samples / frame_step + 1, n_len); i += threadsNum)
            {
                int offset = i * frame_step;

                // apply Hanning window (~10% faster)
                for (int j = 0; j < Math.Min(frame_size, n_samples - offset); j++)
                {
                    fft_in[j] = hann[j] * audio[offset + j];
                }
                // fill the rest with zeros
                if (n_samples - offset < frame_size)
                {
                    //std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
                    float[] temp = new float[frame_size];
                    Array.Copy(fft_in, 0, temp, 0, fft_in.Length);
                    fft_in = temp;
                }

                // FFT
                FFT(fft_in, ref fft_out);

                // Calculate modulus^2 of complex numbers
                // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
                for (int j = 0; j < frame_size; j++)
                {
                    fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
                }

                // mel spectrogram
                for (int j = 0; j < n_mel; j++)
                {
                    double sum = 0.0;
                    // unroll loop (suggested by GH user @lunixbochs)
                    int k = 0;
                    for (k = 0; k < n_fft - 3; k += 4)
                    {
                        sum +=
                                fft_out[k + 0] * _whisperMelFilters.Data[j * n_fft + k + 0] +
                                fft_out[k + 1] * _whisperMelFilters.Data[j * n_fft + k + 1] +
                                fft_out[k + 2] * _whisperMelFilters.Data[j * n_fft + k + 2] +
                                fft_out[k + 3] * _whisperMelFilters.Data[j * n_fft + k + 3];
                    }

                    // handle n_fft remainder
                    for (; k < n_fft; k++)
                    {
                        sum += fft_out[k] * _whisperMelFilters.Data[j * n_fft + k];
                    }
                    sum = Math.Log10(Math.Max(sum, 1e-10));

                    whisperMel.Data[j * n_len + i] = (float)sum;
                }
            }
        }

        public float[] HannWindow(int length, bool periodic)
        {
            float[] output = new float[length];
            int offset = -1;
            if (periodic)
            {
                offset = 0;
            }
            for (int i = 0; i < length; i++)
            {
                output[i] = (float)(0.5 * (1.0 - Math.Cos((2.0 * Math.PI * i) / (length + offset))));
            }
            return output;
        }

        public void FFT(float[] fft_in, ref float[] fft_out)
        {
            //out.resize(in.size() * 2);
            float[] fft_out_temp = new float[fft_in.Length * 2];
            int len_temp = Math.Min(fft_out.Length, fft_out_temp.Length);
            Array.Copy(fft_out, 0, fft_out_temp, 0, len_temp);
            fft_out = fft_out_temp;

            int N = fft_in.Length;

            if (N == 1)
            {
                fft_out[0] = fft_in[0];
                fft_out[1] = 0;
                return;
            }

            if (N % 2 == 1)
            {
                DFT(fft_in, ref fft_out);
                return;
            }

            float[] even = new float[N / 2];
            float[] odd = new float[N / 2];

            //even.reserve(N / 2);
            //odd.reserve(N / 2);
            List<float> evenList = new List<float>();
            List<float> oddList = new List<float>();

            for (int i = 0; i < N; i++)
            {
                if (i % 2 == 0)
                {
                    evenList.Add(fft_in[i]);
                }
                else
                {
                    oddList.Add(fft_in[i]);
                }
            }
            even = evenList.ToArray();
            odd = oddList.ToArray();

            float[] even_fft = new float[0];
            float[] odd_fft = new float[0];

            FFT(even, ref even_fft);
            FFT(odd, ref odd_fft);

            int sin_cos_step = _sinCosNCount / N;
            for (int k = 0; k < N / 2; k++)
            {
                int idx = k * sin_cos_step; // t = 2*M_PI*k/N
                float re = _cosVals[idx]; // cos(t)
                float im = -_sinVals[idx]; // sin(t)

                float re_odd = odd_fft[2 * k + 0];
                float im_odd = odd_fft[2 * k + 1];

                fft_out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
                fft_out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

                fft_out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
                fft_out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
            }
        }

        // naive Discrete Fourier Transform
        // input is real-valued
        // output is complex-valued
        public void DFT(float[] dft_in, ref float[] dft_out)
        {
            int N = dft_in.Length;

            //dft_out.resize(N*2);
            float[] dft_out_temp = new float[N * 2];
            int len_temp = Math.Min(dft_out.Length, dft_out_temp.Length);
            Array.Copy(dft_out, 0, dft_out_temp, 0, len_temp);
            dft_out = dft_out_temp;

            int sin_cos_step = _sinCosNCount / N;

            for (int k = 0; k < N; k++)
            {
                float re = 0;
                float im = 0;

                for (int n = 0; n < N; n++)
                {
                    int idx = (k * n * sin_cos_step) % (_sinCosNCount); // t = 2*M_PI*k*n/N
                    re += dft_in[n] * _cosVals[idx]; // cos(t)
                    im -= dft_in[n] * _sinVals[idx]; // sin(t)
                }

                dft_out[k * 2 + 0] = re;
                dft_out[k * 2 + 1] = im;
            }
        }

        // In FFT, we frequently use sine and cosine operations with the same values.
        // We can use precalculated values to speed up the process.
        public void FillSinCosTable()
        {
            bool is_filled = false;
            if (is_filled) return;
            for (int i = 0; i < _sinCosNCount; i++)
            {
                double theta = (2 * Math.PI * i) / _sinCosNCount;
                _sinVals[i] = (float)Math.Sin(theta);
                _cosVals[i] = (float)Math.Cos(theta);
            }
            is_filled = true;
        }
        public void InitMelFilters(string? melFiltersFilePath = null)
        {
            if (string.IsNullOrEmpty(melFiltersFilePath))
            {
                _whisperMelFilters = new WhisperMelFilters();
                _whisperMelFilters.N_fft = _nFFT;
                _whisperMelFilters.N_mel = _nMels;
                _whisperMelFilters.Data = Filters.GetFilters(_whisperMelFilters.N_mel);
            }
            else
            {
                //Consider implementing reading from a file (mel_filters.npz).
            }

        }

        public void InitState()
        {
            FillSinCosTable();
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_whisperMelFilters != null)
                    {
                        _whisperMelFilters = null;
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
        ~WhisperFeatures()
        {
            Dispose(_disposed);
        }
    }
}
