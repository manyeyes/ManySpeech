// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.MoonshineAsr;

namespace ManySpeech.AliParaformerAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private WavFrontend _wavFrontend;
        private OfflineInputEntity _offlineInputEntity;

        private string? _region;
        private string? _language;
        private List<int> _tokenIds = new List<int>();
        private List<string>? _tokens = new List<string>();
        private List<int[]>? _timestamps = new List<int[]>();
        private List<int[]>? _hotwords = new List<int[]>();

        private static object obj = new object();
        internal OfflineStream(IOfflineProj offlineProj)
        {
            _offlineInputEntity = new OfflineInputEntity();

            _wavFrontend = new WavFrontend(offlineProj.OfflineModel.ConfEntity.frontend_conf, offlineProj.OfflineModel.MvnFilePath);
            _tokenIds = new List<int> { offlineProj.OfflineModel.Blank_id, offlineProj.OfflineModel.Blank_id };
            for (int i = 0; i < _tokenIds.Count; i++)
            {
                _timestamps.Add(new int[2]);
            }
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public List<int> TokenIds { get => _tokenIds; set => _tokenIds = value; }
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<int[]>? Hotwords { get => _hotwords; set => _hotwords = value; }
        public string? Region { get => _region; set => _region = value; }
        public string? Language { get => _language; set => _language = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] fbanks = _wavFrontend.GetFeatures(samples);
                float[] features = _wavFrontend.LfrCmvn(fbanks);
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
                OfflineInputEntity.Hotwords = Hotwords;
            }
        }
        public OfflineInputEntity GetDecodeChunk()
        {
            lock (obj)
            {
                if (OfflineInputEntity.Speech != null && OfflineInputEntity.SpeechLength > 0)
                {
                    OfflineInputEntity.Hotwords = Hotwords;
                }
                return OfflineInputEntity;
            }
        }
        public void RemoveChunk()
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
                    if (_tokenIds != null)
                    {
                        _tokenIds = null;
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
