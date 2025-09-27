// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.WenetAsr.Model;

namespace ManySpeech.WenetAsr
{
    public class OfflineStream:IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private AsrInputEntity _asrInputEntity;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private Int64[] _hyp;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _sampleRate = 16000;
        private int _featureDim = 80;

        private CustomMetadata _customMetadata;
        private List<Int64> _tokens = new List<Int64>();
        private List<int> _timestamps = new List<int>();
        private List<float[]> _states = new List<float[]>();
        private static object obj = new object();
        private int _offset = 0;
        private int _required_cache_size = 0;
        internal OfflineStream(IAsrProj asrProj)
        {
            if (asrProj != null)
            {
                _chunkLength = asrProj.ChunkLength;
                _shiftLength = asrProj.ShiftLength;
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

            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _hyp = new Int64[] { _blank_id, _blank_id };
            _states = GetEncoderInitStates();
            _tokens = new List<Int64> { _blank_id, _blank_id };
        }

        public AsrInputEntity AsrInputEntity { get => _asrInputEntity; set => _asrInputEntity = value; }
        public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = _wavFrontend.GetFbank(samples);
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

        public List<float[]> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> statesList = new List<float[]>();
            //计算尺寸
            int required_cache_size = _required_cache_size < 0 ? 0 : _required_cache_size;
            float[] att_cache = new float[_customMetadata.Num_blocks * _customMetadata.Head * required_cache_size * (_customMetadata.Output_size / _customMetadata.Head * 2)];
            float[] cnn_cache = new float[_customMetadata.Num_blocks * 1 * _customMetadata.Output_size * (_customMetadata.Cnn_module_kernel - 1)];
            statesList.Add(att_cache);
            statesList.Add(cnn_cache);
            return statesList;
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
