// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;

namespace ManySpeech.AliParaformerAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private WavFrontend _wavFrontend;
        private OfflineInputEntity _offlineInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 2;
        private Int64[] _hyp;
        private List<int[]>? _hotwords = new List<int[]>();
        List<Int64> _tokens = new List<Int64>();
        List<int[]> _timestamps = new List<int[]>();
        private static object obj = new object();
        public OfflineStream(string mvnFilePath, ConfEntity confEntity)
        {
            _offlineInputEntity = new OfflineInputEntity();

            _wavFrontend = new WavFrontend(mvnFilePath, confEntity.frontend_conf);
            _hyp = new Int64[] { _blank_id, _blank_id };
            _tokens = new List<Int64> { _blank_id, _blank_id };
            _timestamps = new List<int[]> { };
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public Int64[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<int[]>? Hotwords { get => _hotwords; set => _hotwords = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] fbanks = _wavFrontend.GetFbank(samples);
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
                    if (_hyp != null)
                    {
                        _hyp = null;
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
