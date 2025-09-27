// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.MoonshineAsr.Model;

namespace ManySpeech.MoonshineAsr
{
    public class OnlineStream : IDisposable
    {
        private bool _disposed;
        private AsrInputEntity _asrInputEntity;
        private int _unk_id = 1;
        private int _chunkLength = 0;
        private int _featureDim = 1;
        private List<SegmentEntity> _segments = new List<SegmentEntity>();
        private List<int> _tokens = new List<int>();
        private List<int[]> _timestamps = new List<int[]>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private float[] _cacheSamples = null;
        private int _offset = 0;
        private int _required_cache_size = 0;
        private bool _inputFinished = false;
        internal OnlineStream(IAsrProj asrProj)
        {
            if (asrProj != null)
            {
                _chunkLength = asrProj.ChunkLength;
                _featureDim = asrProj.FeatureDim;
                _required_cache_size = asrProj.Required_cache_size;
                if (_required_cache_size > 0)
                {
                    _offset = _required_cache_size;
                }
            }
            _asrInputEntity = new AsrInputEntity();
            _cacheSamples = new float[320];
            _tokens = new List<int> { _unk_id };
        }

        public AsrInputEntity AsrInputEntity { get => _asrInputEntity; set => _asrInputEntity = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public bool InputFinished { get => _inputFinished; set => _inputFinished = value; }
        public List<SegmentEntity> Segments { get => _segments; set => _segments = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (_cacheSamples.Length > 0)
                {
                    oLen = _cacheSamples.Length;
                }
                float[]? samplesTemp = new float[oLen + samples.Length];
                if (oLen > 0)
                {
                    Array.Copy(_cacheSamples, 0, samplesTemp, 0, oLen);
                }
                Array.Copy(samples, 0, samplesTemp, oLen, samples.Length);
                _cacheSamples = samplesTemp;
                int cacheSamplesLength = _cacheSamples.Length;
                int chunkSamplesLength = _chunkLength;
                if (cacheSamplesLength >= chunkSamplesLength || _inputFinished)
                {
                    //get first segment
                    float[] _samples = new float[chunkSamplesLength];
                    int len = _cacheSamples.Length >= _samples.Length ? _samples.Length : _cacheSamples.Length;
                    Array.Copy(_cacheSamples, 0, _samples, 0, len);
                    if (len < chunkSamplesLength)
                    {
                        _samples = _samples.Select(x => x == 0 ? -23.025850929940457F / 32768 : x).ToArray();
                    }
                    InputSpeech(_samples);
                    //remove first segment
                    float[] _cacheSamplesTemp = new float[cacheSamplesLength - len];
                    Array.Copy(_cacheSamples, len, _cacheSamplesTemp, 0, _cacheSamplesTemp.Length);
                    _cacheSamples = _cacheSamplesTemp;
                }
            }
        }
        public void InputSpeech(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                int oRowLen = 0;
                if (AsrInputEntity.Speech?.Length > 0)
                {
                    oLen = AsrInputEntity.Speech.Length;
                    oRowLen = AsrInputEntity.Speech.Length / _featureDim;
                }
                float[] features = samples;
                int featuresRowLen = features.Length / _featureDim;

                float[]? featuresTemp = new float[oLen + features.Length];
                int featuresTempRowLen = featuresTemp.Length / _featureDim;
                if (AsrInputEntity.SpeechLength > 0)
                {
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(AsrInputEntity.Speech, i * oRowLen, featuresTemp, i * featuresTempRowLen, oRowLen);
                    }
                }
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(features, i * featuresRowLen, featuresTemp, i * featuresTempRowLen + oRowLen, featuresRowLen);
                }
                AsrInputEntity.Speech = featuresTemp;
                AsrInputEntity.SpeechLength = featuresTemp.Length;
            }
        }        

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk(int chunkLength)
        {
            int featureDim = _featureDim;
            lock (obj)
            {
                float[]? decodeChunk = null;
                //use non-streaming asr,get all chunks
                if (chunkLength <= 0)
                {
                    chunkLength = _asrInputEntity.SpeechLength/ featureDim;
                }
                if (chunkLength < _chunkLength)
                {
                    return decodeChunk;
                }
                //use non-streaming asr,get all chunks
                if (chunkLength * featureDim <= _asrInputEntity.SpeechLength)
                {
                    float[] padChunk = new float[chunkLength * featureDim];
                    float[]? features = _asrInputEntity.Speech;
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);
                    decodeChunk = padChunk;
                }
                return decodeChunk;
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = _featureDim;
                if (shiftLength * featureDim <= _asrInputEntity.SpeechLength)
                {
                    float[]? features = _asrInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    _asrInputEntity.Speech = featuresTemp;
                    _asrInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }

        /// <summary>
        /// when is endpoint,determine whether it is completed
        /// </summary>
        /// <param name="isEndpoint"></param>
        /// <returns></returns>
        public bool IsFinished(bool isEndpoint = false)
        {
            int featureDim = _featureDim;
            if (isEndpoint)
            {
                int oLen = 0;
                if (AsrInputEntity.SpeechLength > 0)
                {
                    oLen = AsrInputEntity.SpeechLength;
                }
                if (oLen > 0)
                {
                    var avg = AsrInputEntity.Speech.Average();
                    int num = AsrInputEntity.Speech.Where(x => x != avg).ToArray().Length;
                    if (num == 0)
                    {
                        return true;
                    }
                    else
                    {
                        if (oLen <= _chunkLength * featureDim)
                        {
                            AddSamples(new float[400]);
                        }
                        return false;
                    }
                }
                else
                {
                    return true;
                }
            }
            else
            {
                return false;
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
        ~OnlineStream()
        {
            Dispose(_disposed);
        }
    }
}
