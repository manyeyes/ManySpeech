// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        WhisperFeatures _whisperFeatures;

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _whisperFeatures = new WhisperFeatures(
                nMels: frontendConfEntity.n_mels,
                threadsNum:5,
                melFiltersFilePath:null
                );
        }

        public float[] GetFeatures(float[] samples)
        {
            float[] mel = _whisperFeatures.LogMelSpectrogram(samples);
            return mel;
        }
    
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_whisperFeatures != null)
                {
                    _whisperFeatures.Dispose();
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
