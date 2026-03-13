// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.AudioTagging.Model;

namespace ManySpeech.AudioTagging
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private ModelInputEntity _modelInputEntity;

        private int _sampleRate = 16000;
        private int _featureDim = 80;
        private List<int> _tokens = new List<int>();
        private List<double> _probs = new List<double>();
        private List<string> _taggings=new List<string>();
        private List<int[]> _timestamps = new List<int[]>();
        private List<float[]> _caches = new List<float[]>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private int _offset = 0;
        private int _required_cache_size = 0;
        internal OfflineStream(IOfflineProj offlineProj)
        {
            if (offlineProj != null)
            {
                _featureDim = offlineProj.FeatureDim;
                _sampleRate = offlineProj.SampleRate;
                _required_cache_size = offlineProj.Required_cache_size;
                if (_required_cache_size > 0)
                {
                    _offset = _required_cache_size;
                }
            }

            _modelInputEntity = new ModelInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;

            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _tokens = new List<int> { };
        }

        public ModelInputEntity ModelInputEntity { get => _modelInputEntity; set => _modelInputEntity = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public List<float[]> Caches { get => _caches; set => _caches = value; }
        public List<double> Probs { get => _probs; set => _probs = value; }
        public List<string> Taggings { get => _taggings; set => _taggings = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] waveform = new float[samples.Length + 40];
                Array.Copy(samples, 0, waveform, 0, samples.Length);
                float[] features = _wavFrontend.GetFbank(waveform);
                int oLen = 0;
                if (ModelInputEntity.SpeechLength > 0)
                {
                    oLen = ModelInputEntity.SpeechLength;
                }
                float[]? featuresTemp = new float[oLen + features.Length];
                if (ModelInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_modelInputEntity.Speech, 0, featuresTemp, 0, _modelInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, ModelInputEntity.SpeechLength, features.Length);
                ModelInputEntity.Speech = featuresTemp;
                ModelInputEntity.SpeechLength = featuresTemp.Length;
            }
        }
        public float[]? GetDecodeChunk()
        {
            lock (obj)
            {
                float[]? decodeChunk = null;
                decodeChunk = ModelInputEntity.Speech;
                return decodeChunk;
            }
        }
        public void RemoveDecodedChunk()
        {
            lock (obj)
            {
                if (_tokens.Count > 2)
                {
                    ModelInputEntity.Speech = null;
                    ModelInputEntity.SpeechLength = 0;
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
