// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.FireRedAsr.Model;

namespace ManySpeech.FireRedAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private AsrInputEntity _asrInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _pad_id = 2;
        private int _sos_id = 3;
        private int _eos_id = 4;
        //private Int64[] _hyp;
        private int _sampleRate = 16000;
        private int _featureDim = 80;

        private CustomMetadata _customMetadata;
        private List<Int64> _tokens = new List<Int64>();
        private List<int> _timestamps = new List<int>();
        private List<float[]> _caches = new List<float[]>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private int _offset = 0;
        private int _required_cache_size = 0;
        internal OfflineStream(string mvnFilePath, IAsrProj asrProj)
        {
            if (asrProj != null)
            {
                _featureDim = asrProj.FeatureDim;
                _sampleRate = asrProj.SampleRate;
                _customMetadata = asrProj.CustomMetadata;
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

            _wavFrontend = new WavFrontend(mvnFilePath,_frontendConfEntity);
            //_hyp = new Int64[] { _sos_id };
            _caches = GetDecoderInitCaches();
            _states = GetDecoderInitCaches();
            _tokens = new List<Int64> { _sos_id };
        }

        public AsrInputEntity AsrInputEntity { get => _asrInputEntity; set => _asrInputEntity = value; }
        //public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public List<float[]> Caches { get => _caches; set => _caches = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = _wavFrontend.GetFbank(samples);
                features = _wavFrontend.ApplyCmvn(features);
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

        public List<float[]> GetDecoderInitCaches(int batchSize = 1)
        {
            float[] cache = new float[0];
            List<float[]> cachesList = new List<float[]>();
            for (int i = 0; i < 16; i++)
            {
                cachesList.Add(cache);
            }
            //float[] caches = cacheList
            //    .Where(a => a != null)  // 过滤掉 null 子数组
            //    .SelectMany(a => a)
            //    .ToArray();
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
