// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.DolphinAsr.Model;
using SpeechFeatures;

namespace ManySpeech.DolphinAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private OnlineFbank? _onlineFbank;
        private int _sampleRate = 16000;
        private int _speechLength = 30;
        private bool _isPaddingSpeech = false;

        public WavFrontend(FrontendConfig? frontendConfig = null, int sampleRate = 16000, int speechLength = 30, bool isPaddingSpeech = false)
        {
            if (frontendConfig != null)
            {
                _onlineFbank = new OnlineFbank(
                    dither: frontendConfig.dither,
                    snip_edges: frontendConfig.snip_edges,
                    sample_rate: frontendConfig.fs,
                    num_bins: frontendConfig.n_mels,
                    window_type: frontendConfig.window,
                    frame_length: frontendConfig.frame_length,
                    frame_shift: frontendConfig.frame_shift,
                    is_librosa: frontendConfig.is_librosa,
                    htk_mode: frontendConfig.htk_mode,
                    low_freq: frontendConfig.low_freq,
                    norm: frontendConfig.norm,
                    remove_dc_offset: frontendConfig.remove_dc_offset,
                    preemph_coeff: frontendConfig.preemph_coeff,
                    use_log_fbank: frontendConfig.use_log_fbank
                    );
            }
            _sampleRate = sampleRate;
            _speechLength = speechLength;
            _isPaddingSpeech = isPaddingSpeech;
        }

        public float[] GetFeatures(float[] samples)
        {
            float[] features = _isPaddingSpeech ? ResizeAudioDuration(samples, _sampleRate, _speechLength) : samples;
            if (_onlineFbank != null)
            {
                features = _onlineFbank.GetFbank(features);
            }
            return features;
        }
        /// <summary>
        /// Resamples and pads/truncates raw audio data to a fixed length based on target speech duration.
        /// (Resamples and pads/truncates raw audio data to a fixed length according to the target speech duration)
        /// </summary>
        /// <param name="raw">Raw audio PCM data in float format (16-bit PCM normalized to [-1.0, 1.0])</param>
        /// <param name="sampleRate">Audio sample rate in Hz (e.g., 16000, 44100)</param>
        /// <param name="speechLength">Target speech duration in seconds (0 = return original data)</param>
        /// <returns>Normalized audio data with fixed length (padded with 0s or truncated)</returns>
        public float[] ResizeAudioDuration(float[] raw, float sampleRate, float speechLength)
        {
            // Return original data if target duration is 0 (no resizing needed)
            if (speechLength == 0) return raw;

            // Calculate target number of samples based on sample rate and duration
            int targetSampleCount = (int)(sampleRate * speechLength);
            float[] processedAudio;

            if (raw.Length >= targetSampleCount)
            {
                // Truncate to target length - take first N samples
                processedAudio = new float[targetSampleCount];
                Array.Copy(raw, 0, processedAudio, 0, targetSampleCount);
            }
            else
            {
                // Pad with zeros to reach target length - copy original data to start, rest filled with 0.0f
                processedAudio = new float[targetSampleCount];
                Array.Copy(raw, 0, processedAudio, 0, raw.Length);
                // Remaining elements in new array are already initialized to 0.0f by default
            }

            return processedAudio;
        }
        public void InputFinished()
        {
            _onlineFbank.InputFinished();
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
