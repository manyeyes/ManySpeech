// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;

namespace ManySpeech.K2TransducerAsr
{
    public class OnlineStream : IDisposable
    {
        private bool _disposed;

        private WavFrontend? _wavFrontend;
        private OnlineInputEntity? _onlineInputEntity;
        private Int64[]? _hyp;
        private List<Int64>? _tokens = new List<Int64>();
        private List<int>? _timestamps = new List<int>();
        private List<List<float[]>>? _states = new List<List<float[]>>();
        private int _frameOffset = 0;
        private int _numTrailingBlank = 0;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _sampleRate = 16000;
        private int _featureDim = 80;
        private static object obj = new object();
        internal OnlineStream(IOnlineProj? onlineProj)
        {
            if (onlineProj != null)
            {
                _states = onlineProj.GetEncoderInitStates();
                _chunkLength = onlineProj.ChunkLength;
                _shiftLength = onlineProj.ShiftLength;
                _featureDim = onlineProj.FeatureDim;
                _sampleRate = onlineProj.SampleRate;
                _onlineInputEntity = new OnlineInputEntity();
                FrontendConfEntity frontendConfEntity = new FrontendConfEntity();
                frontendConfEntity.fs = _sampleRate;
                frontendConfEntity.n_mels = _featureDim;
                frontendConfEntity.feature_type = onlineProj.CustomMetadata.Feature_type;
                if (onlineProj.CustomMetadata.Feature_type == "whisper")
                {
                    frontendConfEntity.window = "hanning";
                    frontendConfEntity.n_mels = 80;
                    frontendConfEntity.snip_edges = false;
                }
                _wavFrontend = new WavFrontend(frontendConfEntity);
                int blank_id = onlineProj.Blank_id;
                _hyp = new Int64[] { blank_id, blank_id };
                _tokens = new List<Int64> { blank_id, blank_id };
            }
        }

        public OnlineInputEntity? OnlineInputEntity { get => _onlineInputEntity; set => _onlineInputEntity = value; }
        public long[]? Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64>? Tokens { get => _tokens; set => _tokens = value; }
        public List<int>? Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<List<float[]>>? States { get => _states; set => _states = value; }
        public int FrameOffset { get => _frameOffset; set => _frameOffset = value; }
        public int NumTrailingBlank { get => _numTrailingBlank; set => _numTrailingBlank = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (OnlineInputEntity?.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                float[]? features = _wavFrontend?.GetFbank(samples);
                if (features?.Length > 0)
                {
                    float[]? featuresTemp = new float[oLen + features.Length];
                    if (OnlineInputEntity?.Speech != null && OnlineInputEntity.SpeechLength > 0)
                    {
                        Array.Copy(OnlineInputEntity.Speech, 0, featuresTemp, 0, OnlineInputEntity.SpeechLength);
                    }
                    Array.Copy(features, 0, featuresTemp, OnlineInputEntity.SpeechLength, features.Length);
                    OnlineInputEntity.Speech = featuresTemp;
                    OnlineInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk()
        {
            int chunkLength = _chunkLength;
            int featureDim = _featureDim;
            lock (obj)
            {
                float[]? padChunk = new float[chunkLength * featureDim];
                if (chunkLength * featureDim <= OnlineInputEntity?.SpeechLength)
                {
                    float[]? features = OnlineInputEntity.Speech;
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);
                    return padChunk;
                }
                else
                {
                    return null;
                }
            }
        }

        public void RemoveChunk()
        {
            int shiftLength = _shiftLength;
            lock (obj)
            {
                int featureDim = _featureDim;
                if (shiftLength * featureDim <= OnlineInputEntity?.SpeechLength)
                {
                    float[]? features = OnlineInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    OnlineInputEntity.Speech = featuresTemp;
                    OnlineInputEntity.SpeechLength = featuresTemp.Length;
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
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                if (oLen > 0)
                {
                    var avg = OnlineInputEntity.Speech.Average();
                    int num = OnlineInputEntity.Speech.Where(x => x != avg).ToArray().Length;
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
                    if (_wavFrontend != null)
                    {
                        _wavFrontend.Dispose();
                    }
                    if (_onlineInputEntity != null)
                    {
                        _onlineInputEntity = null;
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
                    if (_states != null)
                    {
                        _states = null;
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
