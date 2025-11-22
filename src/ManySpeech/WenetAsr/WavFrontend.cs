// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using SpeechFeatures;
using System.Data;
using ManySpeech.WenetAsr.Model;

namespace ManySpeech.WenetAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        OnlineFbank _onlineFbank;

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                window_type:_frontendConfEntity.window
                );
        }

        public float[] GetFbank(float[] samples)
        {
            samples = samples.Select(x => (float)Float32ToInt16(x)).ToArray();
            float[] fbanks = _onlineFbank.GetFbank(samples);
            return fbanks;
        }
        public void InputFinished()
        {
            _onlineFbank.InputFinished();
        }
        private static Int16 Float32ToInt16(float sample)
        {
            if (sample < -0.999999f)
            {
                return Int16.MinValue;
            }
            else if (sample > 0.999999f)
            {
                return Int16.MaxValue;
            }
            else
            {
                if (sample < 0)
                {
                    return (Int16)(Math.Floor(sample * 32767.0f));
                }
                else
                {
                    return (Int16)(Math.Ceiling(sample * 32767.0f));
                }
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
