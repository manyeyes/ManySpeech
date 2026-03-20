// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.DolphinAsr.Model;

namespace ManySpeech.DolphinAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private FrontendConfig? _frontendConfig;
        private WavFrontend _wavFrontend;
        private OfflineInputEntity _offlineInputEntity;
        private CustomMetadata _customMetadata;

        private string? _region;
        private string? _language;
        private List<int> _tokenIds = new List<int>();
        private List<string>? _tokens = new List<string>();
        private List<int[]>? _timestamps = new List<int[]>();

        private static object obj = new object();
        private int _offset = 0;
        private int _requiredCacheSize = 0;
        private List<float[]> _caches = new List<float[]>();
        private List<float[]> _states = new List<float[]>();

        internal OfflineStream(IOfflineProj offlineProj)
        {
            if (offlineProj != null)
            {
                _customMetadata = offlineProj.OfflineModel.CustomMetadata;
                _requiredCacheSize = offlineProj.OfflineModel.RequiredCacheSize;
                if (_requiredCacheSize > 0)
                {
                    _offset = _requiredCacheSize;
                }
            }

            _offlineInputEntity = new OfflineInputEntity();
            if (offlineProj.OfflineModel.ConfEntity.preprocessor_conf.use_wavfrontend)
            {
                _frontendConfig = offlineProj.OfflineModel.ConfEntity.frontend_conf;
            }
            _wavFrontend = new WavFrontend(frontendConfig: _frontendConfig, sampleRate: offlineProj.OfflineModel.SampleRate, speechLength: offlineProj.OfflineModel.SpeechLength);
            _caches = GetDecoderInitCaches();
            _states = GetDecoderInitCaches();
            _tokenIds = new List<int> { offlineProj.OfflineModel.SosId };
            for (int i = 0; i < _tokenIds.Count; i++)
            {
                _timestamps.Add(new int[] { 0, 0 });
            }
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public string? Region { get => _region; set => _region = value; }
        public string? Language { get => _language; set => _language = value; }
        public List<int> TokenIds { get => _tokenIds; set => _tokenIds = value; }
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public List<float[]> Caches { get => _caches; set => _caches = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] waveform = new float[samples.Length];
                Array.Copy(samples, 0, waveform, 0, samples.Length);
                float[] features = _wavFrontend.GetFeatures(waveform);
                int oLen = 0;
                if (OfflineInputEntity.SpeechLength > 0)
                {
                    oLen = OfflineInputEntity.SpeechLength;
                }
                float[]? featuresTemp = new float[oLen + features.Length];
                if (OfflineInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_offlineInputEntity.Speech, 0, featuresTemp, 0, _offlineInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, OfflineInputEntity.SpeechLength, features.Length);
                OfflineInputEntity.Speech = featuresTemp;
                OfflineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }
        public float[]? GetDecodeChunk()
        {
            lock (obj)
            {
                float[]? decodeChunk = null;
                decodeChunk = OfflineInputEntity.Speech;
                return decodeChunk;
            }
        }
        public void RemoveDecodedChunk()
        {
            lock (obj)
            {
                if (_tokens.Count > 2)
                {
                    OfflineInputEntity.Speech = null;
                    OfflineInputEntity.SpeechLength = 0;
                }
            }
        }

        public List<float[]> GetDecoderInitCaches(int batchSize = 1)
        {
            float[] cache = new float[0];
            List<float[]> cachesList = new List<float[]>();
            for (int i = 0; i < 6; i++)
            {
                cachesList.Add(cache);
            }
            return cachesList;
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
                    if (_tokens != null)
                    {
                        _tokens = null;
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
