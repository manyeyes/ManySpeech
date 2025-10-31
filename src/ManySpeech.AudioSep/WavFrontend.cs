// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.AudioSep.Model;
using SpeechFeatures;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// WavFrontend
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        private OnlineFbank _onlineFbank;
        private const double EPS = 1e-6; // 定义 EPS 常量，用于数值稳定性

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                window_type: _frontendConfEntity.window,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                frame_shift: _frontendConfEntity.frame_shift,
                frame_length: _frontendConfEntity.frame_length
                );
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            float[] fbanks = _onlineFbank.GetFbank(samples);
            return fbanks;
        }

        
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_onlineFbank != null)
                {
                    _onlineFbank.Dispose();
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
