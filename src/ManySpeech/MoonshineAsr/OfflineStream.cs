// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.MoonshineAsr.Model;

namespace ManySpeech.MoonshineAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        private AsrInputEntity _asrInputEntity;
        private int _unk_id = 1;
        private int _sampleRate = 16000;
        private int _featureDim = 80;

        private List<int> _tokens = new List<int>();
        private List<int[]> _timestamps = new List<int[]>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private int _offset = 0;
        private int _required_cache_size = 0;
        internal OfflineStream(IAsrProj asrProj)
        {
            if (asrProj != null)
            {
                _featureDim = asrProj.FeatureDim;
                _sampleRate = asrProj.SampleRate;
                _required_cache_size = asrProj.Required_cache_size;
                if (_required_cache_size > 0)
                {
                    _offset = _required_cache_size;
                }
            }

            _asrInputEntity = new AsrInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;
            _tokens = new List<int> { _unk_id };
        }

        public AsrInputEntity AsrInputEntity { get => _asrInputEntity; set => _asrInputEntity = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = samples;
                int oLen = 0;
                if (AsrInputEntity.SpeechLength > 0)
                {
                    oLen = AsrInputEntity.SpeechLength;
                }
                float[]? featuresTemp = new float[oLen + features.Length];
                if (AsrInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_asrInputEntity.Speech, 0, featuresTemp, 0, _asrInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, AsrInputEntity.SpeechLength, features.Length);
                AsrInputEntity.Speech = featuresTemp;
                AsrInputEntity.SpeechLength = featuresTemp.Length;
            }
        }
        public float[]? GetDecodeChunk()
        {
            lock (obj)
            {
                float[]? decodeChunk = null;
                decodeChunk = AsrInputEntity.Speech;
                return decodeChunk;
            }
        }
        public void RemoveDecodedChunk()
        {
            lock (obj)
            {
                if (_tokens.Count > 2)
                {
                    AsrInputEntity.Speech = null;
                    AsrInputEntity.SpeechLength = 0;
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_asrInputEntity != null)
                    {
                        _asrInputEntity = null;
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
