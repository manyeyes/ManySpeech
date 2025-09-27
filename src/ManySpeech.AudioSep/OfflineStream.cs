// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.AudioSep.Model;

namespace ManySpeech.AudioSep
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private string _audioId = string.Empty;
        private List<ModelOutputEntity> _modelOutputEntities;
        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private ModelInputEntity _modelInputEntity;

        private CustomMetadata _customMetadata;
        private List<Int64> _tokens = new List<Int64>();
        private List<int> _timestamps = new List<int>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private int _offset = 0;

        private int _channels = 1;
        private int _sampleRate = 16000;
        internal OfflineStream(string mvnFilePath, ISepProj asrProj)
        {
            if (asrProj != null)
            {
                _channels = asrProj.Channels;
                _sampleRate = asrProj.SampleRate;
            }

            _modelInputEntity = new ModelInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _wavFrontend = new WavFrontend(_frontendConfEntity);
        }

        public ModelInputEntity ModelInputEntity { get => _modelInputEntity; set => _modelInputEntity = value; }
        public string AudioId { get => _audioId; set => _audioId = value; }
        public List<ModelOutputEntity> ModelOutputEntities { get => _modelOutputEntities; set => _modelOutputEntities = value; }
        public int Channels { get => _channels; set => _channels = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = samples;
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
                if (ModelInputEntity != null)
                {
                    ModelInputEntity.Speech = null;
                    ModelInputEntity.SpeechLength = 0;
                }
            }
        }

        public void RemoveSamples()
        {
            lock (obj)
            {
                if (ModelInputEntity != null)
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
                    if (_modelOutputEntities != null)
                    {
                        _modelOutputEntities = null;
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
