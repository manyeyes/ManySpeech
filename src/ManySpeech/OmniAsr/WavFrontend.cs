// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.OmniAsr.Model;
using SpeechFeatures;
//using KaldiNativeFbankSharp;

namespace ManySpeech.OmniAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    internal class WavFrontend : IDisposable
    {
        private bool _disposed;
        public WavFrontend()
        {

        }

        public float[] GetFeatures(float[] samples)
        {
            if (samples == null) return null;
            float maxAbs = samples.Max(x => Math.Abs(x));
            if (maxAbs <= 0) return samples.ToArray();
            // 归一化
            float scale = 0.9f / maxAbs;
            float[] result = new float[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                result[i] = samples[i] * scale;
            }
            return samples;
        }

        public void InputFinished()
        {
            //
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    //
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
