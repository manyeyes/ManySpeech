// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.AudioTagging.Model;
using SpeechFeatures;

namespace ManySpeech.AudioTagging
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        // N_MELS
        private int _nMels = 64;          // Number of mel bands
        // TARGET_LENGTH
        private int _targetLength = 1012;   // Target number of time frames

        private FrontendConfEntity _frontendConfEntity;
        private OnlineFbank _onlineFbank;

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                window_type: _frontendConfEntity.window,
                frame_length: _frontendConfEntity.frame_length,
                frame_shift: _frontendConfEntity.frame_shift,
                is_librosa: _frontendConfEntity.is_librosa,
                htk_mode: _frontendConfEntity.htk_mode,
                low_freq: _frontendConfEntity.low_freq,
                norm: _frontendConfEntity.norm,
                remove_dc_offset: _frontendConfEntity.remove_dc_offset,
                preemph_coeff: _frontendConfEntity.preemph_coeff,
                use_log_fbank: _frontendConfEntity.use_log_fbank
                );
            _nMels = _frontendConfEntity.n_mels;
        }

        public float[] GetFbank(float[] samples)
        {
            float[] fbanks = _onlineFbank.GetFbank(samples);
            fbanks = GetMelDb(fbanks);
            return fbanks;
        }
        public void InputFinished()
        {
            _onlineFbank.InputFinished();
        }

        /// <summary>
        /// Generates dB-scaled mel spectrogram from flattened mel spectrum data,
        /// unifies the time dimension, adds a batch dimension, and returns as a flattened array.
        /// </summary>
        /// <param name="features">Flattened 1D array stored in time-major order, shape (time, N_MELS).</param>
        /// <returns>Flattened 1D array with shape (1, N_MELS, TARGET_LENGTH) stored in row-major order (mel bins first, then time).</returns>
        public float[] GetMelDb(float[] features)
        {
            int totalOutputLength = _nMels * _targetLength;

            // Handle empty input or insufficient length
            if (features == null || features.Length == 0)
            {
                return new float[totalOutputLength];
            }

            // Calculate number of time frames (assuming length is a multiple of N_MELS)
            int totalLength = features.Length;
            int timeFrames = totalLength / _nMels;
            if (totalLength % _nMels != 0)
            {
                // If length is not a multiple, truncate to the largest integer multiple (can be changed to throw exception if needed)
                timeFrames = totalLength / _nMels;
                // Optional: add log warning here
            }

            // Convert the flattened input array to a 2D array melStack[N_MELS, timeFrames]
            // Input index: features[t * N_MELS + m] corresponds to the m-th mel value of the t-th frame
            float[,] melStack = new float[_nMels, timeFrames];
            for (int t = 0; t < timeFrames; t++)
            {
                for (int m = 0; m < _nMels; m++)
                {
                    int inputIndex = t * _nMels + m;          // time-major order
                    melStack[m, t] = features[inputIndex];    // transpose to mel-major order
                }
            }

            // Convert to dB scale
            float[,] melDb = AmpToDb(melStack);

            // Adjust time dimension to TARGET_LENGTH and directly fill into the final 1D array
            float[] result = new float[totalOutputLength];

            if (timeFrames >= _targetLength)
            {
                // Truncate: take the first TARGET_LENGTH columns
                for (int m = 0; m < _nMels; m++)
                {
                    for (int t = 0; t < _targetLength; t++)
                    {
                        // Output index: mel bins first, then time (batch=0)
                        int outputIndex = m * _targetLength + t;
                        result[outputIndex] = melDb[m, t];
                    }
                }
            }
            else
            {
                // Pad: copy existing data, remaining columns remain zero by default
                for (int m = 0; m < _nMels; m++)
                {
                    for (int t = 0; t < timeFrames; t++)
                    {
                        int outputIndex = m * _targetLength + t;
                        result[outputIndex] = melDb[m, t];
                    }
                    // Remaining columns are already initialized to 0 when the array was created
                }
            }

            return result;
        }

        /// <summary>
        /// Converts linear magnitude spectrum to dB scale with dynamic range clipping.
        /// </summary>
        /// <param name="x">Linear magnitude spectrum, 2D array (N_MELS, time).</param>
        /// <param name="topDb">Upper bound of dynamic range (attenuation relative to global maximum), default 120.0.</param>
        /// <returns>dB-scaled spectrum with the same shape as x.</returns>
        private float[,] AmpToDb(float[,] x, float topDb = 120.0f)
        {
            int rows = x.GetLength(0);
            int cols = x.GetLength(1);
            float[,] db = new float[rows, cols];

            const float minAmp = 1e-10f;   // avoid log10(0)
            float maxDb = float.NegativeInfinity;

            // First pass: compute dB and find global maximum
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float safeAmp = x[i, j] > minAmp ? x[i, j] : minAmp;
                    float val = 10.0f * (float)Math.Log10(safeAmp);
                    db[i, j] = val;
                    if (val > maxDb)
                        maxDb = val;
                }
            }

            // Second pass: clip values below (maxDb - topDb)
            float lowerBound = maxDb - topDb;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (db[i, j] < lowerBound)
                        db[i, j] = lowerBound;
                }
            }

            return db;
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
