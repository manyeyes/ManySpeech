// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;

namespace ManySpeech.K2TransducerAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private WavFrontend _wavFrontend;
        private OfflineInputEntity _offlineInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 2;
        private OfflineCustomMetadata _offlineCustomMetadata;
        List<Int64> _tokens = new List<Int64>();
        List<int> _timestamps = new List<int>();
        private static object obj = new object();
        private int _frameOffset = 0;
        private int _numTrailingBlank = 0;
        public OfflineStream(OfflineCustomMetadata offlineCustomMetadata, int sampleRate = 16000, int featureDim = 80)
        {
            _offlineCustomMetadata = offlineCustomMetadata;
            _offlineInputEntity = new OfflineInputEntity();
            FrontendConfEntity frontendConfEntity = new FrontendConfEntity();
            frontendConfEntity.fs = sampleRate;
            frontendConfEntity.n_mels = featureDim;
            frontendConfEntity.feature_type=offlineCustomMetadata.Feature_type;
            if (offlineCustomMetadata.Feature_type == "whisper")
            {
                frontendConfEntity.window = "hanning";
                frontendConfEntity.n_mels = 80;
                frontendConfEntity.snip_edges = false;
            }
            _wavFrontend = new WavFrontend(frontendConfEntity);
            _tokens = new List<Int64> { _blank_id, _blank_id };
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public int FrameOffset { get => _frameOffset; set => _frameOffset = value; }
        public int NumTrailingBlank { get => _numTrailingBlank; set => _numTrailingBlank = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = _wavFrontend.GetFbank(samples);
                float[]? featuresTemp = new float[OfflineInputEntity.SpeechLength + features.Length];
                if (OfflineInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_offlineInputEntity.Speech, 0, featuresTemp, 0, _offlineInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, OfflineInputEntity.SpeechLength, features.Length);
                OfflineInputEntity.Speech = featuresTemp;
                OfflineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }
        public void RemoveSamples()
        {
            lock (obj)
            {
                if (_tokens.Count > _offlineCustomMetadata.Context_size)
                {
                    OfflineInputEntity.Speech = null;
                    OfflineInputEntity.SpeechLength = 0;
                }
            }
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_wavFrontend != null)
                    {
                        _wavFrontend.Dispose();
                    }
                    if (_offlineInputEntity != null)
                    {
                        _offlineInputEntity = null;
                    }
                    if (_tokens != null)
                    {
                        _tokens = null;
                    }
                    if (_timestamps != null)
                    {
                        _timestamps = null;
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
        ~OfflineStream()
        {
            Dispose(_disposed);
        }        
    }
}
