// See https://github.com/manyeyes for more information
// Copyright (c) 2025 by manyeyes
using ManySpeech.AudioSep.Model;
using SpeechFeatures;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// Handles audio frontend processing to extract filter bank features
    /// </summary>
    internal class WavFrontend : IDisposable
    {
        private readonly FrontendConfEntity _frontendConfig;
        private readonly OnlineFbank _onlineFbank;
        private const double EPS = 1e-6; // Constant for numerical stability

        /// <summary>
        /// Initializes a new instance of the WavFrontend class
        /// </summary>
        /// <param name="frontendConfig">Configuration for frontend processing</param>
        public WavFrontend(FrontendConfEntity frontendConfig)
        {
            _frontendConfig = frontendConfig;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfig.dither,
                snip_edges: _frontendConfig.snip_edges,
                window_type: _frontendConfig.window,
                sample_rate: _frontendConfig.fs,
                num_bins: _frontendConfig.n_mels,
                frame_shift: _frontendConfig.frame_shift,
                frame_length: _frontendConfig.frame_length
            );
        }

        /// <summary>
        /// Extracts filter bank features from audio samples
        /// </summary>
        /// <param name="samples">Input audio samples</param>
        /// <returns>Extracted filter bank bank features as float array</returns>
        public float[] GetFbank(float[] samples)
        {
            return _onlineFbank.GetFbank(samples);
        }

        /// <summary>
        /// Releases all resources used by the WavFrontend instance
        /// </summary>
        /// <param name="disposing">True if called from managed code, false if from finalizer</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _onlineFbank?.Dispose();
                // FrontendConfEntity is a POCO and doesn't need explicit nulling
            }
        }

        /// <summary>
        /// Releases the WavFrontend instance and releases all resources
        /// </summary>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}