// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.OmniAsr.Model;

namespace ManySpeech.OmniAsr
{
    public class OfflineStream : IDisposable
    {
        private bool _disposed;

        private WavFrontend _wavFrontend;
        private OfflineInputEntity _offlineInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 3;
        List<int> _tokens = new List<int>();
        List<int[]> _timestamps = new List<int[]>();
        private static object obj = new object();
        private int _frameOffset = 0;
        private int _numTrailingBlank = 0;
        private int _contextSize = 2;
        public OfflineStream(IOfflineProj offlineProj)
        {
            _offlineInputEntity = new OfflineInputEntity();
            _wavFrontend = new WavFrontend();
            _tokens = new List<int> { _blank_id, _blank_id };
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public int FrameOffset { get => _frameOffset; set => _frameOffset = value; }
        public int NumTrailingBlank { get => _numTrailingBlank; set => _numTrailingBlank = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] waveform = new float[samples.Length + 40];
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
