// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    public class OnlineStream : IDisposable
    {
        private bool _disposed;
        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private OnlineInputEntity _onlineInputEntity;
        //private Int64[] _hyp;
        private int _chunkLength = 0;
        private int _frameLength = 0;
        private int _shiftLength = 0;
        private int _hopLength = 0;
        private int _sampleRate = 16000;
        private int _featureDim = 80;
        private float _suppressSample = float.NaN;

        private CustomMetadata? _customMetadata;
        private List<int> _tokens = new List<int>();
        //private string? _language;
        //private List<int[]> _timestamps = new List<int[]>();
        private int _seek = 0;
        //private List<float[]> _states = new List<float[]>();
        //private List<int> _all_tokens = new List<int>();
        private List<SegmentEntity> _allSegments = new List<SegmentEntity>();
        private int _startIdx = 0;
        private static object obj = new object();
        private float[] _cacheSamples = null;
        private int _offset = 0;
        private int _prompt_reset_since = 0;
        private List<int> _decodingPrompt = null;
        private bool _inputFinished = false;
        internal OnlineStream(OnlineModel? onlineModel)
        {
            if (onlineModel != null)
            {
                _chunkLength = onlineModel.ChunkLength;
                _frameLength = onlineModel.FrameLength;
                _shiftLength = onlineModel.ShiftLength;
                _hopLength = onlineModel.HopLength;
                _featureDim = onlineModel.FeatureDim;
                _sampleRate = onlineModel.SampleRate;
                _suppressSample = onlineModel.SuppressSample;
                _customMetadata = onlineModel.CustomMetadata;
            }

            _onlineInputEntity = new OnlineInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;

            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _cacheSamples = new float[0];//new float[_hopLength * _frameLength];//new float[160 * 3000];
            _tokens = new List<int> { };
        }

        public OnlineInputEntity OnlineInputEntity { get => _onlineInputEntity; set => _onlineInputEntity = value; }
        //public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        //public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        //public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        //public List<int> All_tokens { get => _all_tokens; set => _all_tokens = value; }
        public List<SegmentEntity> AllSegments { get => _allSegments; set => _allSegments = value; }
        //public string? Language { get => _language; set => _language = value; }
        public int Seek { get => _seek; set => _seek = value; }
        public bool InputFinished { get => _inputFinished; set => _inputFinished = value; }
        public int Prompt_reset_since { get => _prompt_reset_since; set => _prompt_reset_since = value; }
        public List<int> DecodingPrompt { get => _decodingPrompt; set => _decodingPrompt = value; }

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
                int chunkSamplesLength = _hopLength * _frameLength/30*5;//160 * 1000;// onlinemodel.frameLength
                if (cacheSamplesLength >= chunkSamplesLength || InputFinished)
                {
                    //get first segment
                    float[] _samples = new float[chunkSamplesLength];
                    //float epsilon = 1e-3f;
                    //// 填充极小值（可选择交替符号避免直流偏移）
                    //for (int i = 0; i < _samples.Length; i++)
                    //{
                    //    // 每隔一个采样点取反，减少直流分量
                    //    _samples[i] = (i % 2 == 0) ? epsilon : -epsilon;
                    //}
                    int len = _cacheSamples.Length >= _samples.Length ? _samples.Length : _cacheSamples.Length;
                    Array.Copy(_cacheSamples, 0, _samples, 0, len);
                    if (len < chunkSamplesLength)
                    {
                        _samples = _samples.Select(x => x == 0 ? -0.00070269317413142F : x).ToArray();
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
                if (OnlineInputEntity.Speech?.Length > 0)
                {
                    oLen = OnlineInputEntity.Speech.Length;
                    oRowLen = OnlineInputEntity.Speech.Length / _featureDim;
                }
                float[] features = _wavFrontend.GetFeatures(samples);
                int featuresRowLen = features.Length / _featureDim;

                float[]? featuresTemp = new float[oLen + features.Length];//new matrix
                int featuresTempRowLen = featuresTemp.Length / _featureDim;
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(_onlineInputEntity.Speech, i * oRowLen, featuresTemp, i * featuresTempRowLen, oRowLen);
                    }
                }
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(features, i * featuresRowLen, featuresTemp, i * featuresTempRowLen + oRowLen, featuresRowLen);
                }
                OnlineInputEntity.Speech = featuresTemp;
                OnlineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk()//nFrames
        {
            lock (obj)
            {
                float[]? decodeChunk = null;
                if (_onlineInputEntity.Speech == null)
                {
                    return decodeChunk;
                }
                if (_onlineInputEntity.Speech?.Length / _featureDim < _frameLength)
                {
                    if (_inputFinished)
                    {
                        decodeChunk = new float[_frameLength * _featureDim];
                        float[]? features = _onlineInputEntity.Speech;
                        int oRowLen = _onlineInputEntity.Speech.Length / _featureDim;
                        for (int i = 0; i < _featureDim; i++)
                        {
                            Array.Copy(features, i * oRowLen, decodeChunk, i * _frameLength, oRowLen);
                        }
                    }
                    else
                    {
                        return decodeChunk;
                    }
                }
                else
                {
                    //// use non-streaming asr, get all chunks
                    // 3000 frames
                    decodeChunk = new float[_frameLength * _featureDim];
                    float[]? features = _onlineInputEntity.Speech;
                    int oRowLen = _onlineInputEntity.Speech.Length / _featureDim;
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(features, i * oRowLen, decodeChunk, i * _frameLength, _frameLength);
                    }
                }
                //dim min head length : 398
                float[] firstRowChunk = new float[_frameLength - 1];
                Array.Copy(decodeChunk, 0, firstRowChunk, 0, firstRowChunk.Length);
                var firstRowAvg = firstRowChunk.Average();
                int firstRowAvgNum = firstRowChunk.Where(x => x == firstRowAvg).ToArray().Length;
                //dim min head length : 398
                float[] headChunk = new float[398 * _featureDim];
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(decodeChunk, i * _frameLength, headChunk, i * 398, 398);
                }
                var headAvg = headChunk.Average();
                int headAvgNum = headChunk.Where(x => x == headAvg).ToArray().Length;
                //dim min tail length : 398
                float[] tailChunk = new float[398 * _featureDim];
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(decodeChunk, i * _frameLength + _frameLength - 400, tailChunk, i * 398, 398);
                }
                var tailAvg = tailChunk.Average();
                int tailAvgNum = tailChunk.Where(x => x == tailAvg).ToArray().Length;
                if (firstRowAvgNum == firstRowChunk.Length || headAvgNum == headChunk.Length)// || tailAvgNum == 0
                {
                    decodeChunk = decodeChunk.Select(x => _suppressSample).ToArray();
                }
                else if (tailAvgNum == tailChunk.Length)
                {
                    int len = 0;
                    for (int i = decodeChunk.Length / _featureDim - 1; i >= 0; i--)
                    {
                        if (decodeChunk[i] == tailAvg)
                        {
                            len++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    float[] tempChunk = new float[len];
                    //float epsilon = 1e-3f;
                    //// 填充极小值（可选择交替符号避免直流偏移）
                    //for (int i = 0; i < tempChunk.Length; i++)
                    //{
                    //    // 每隔一个采样点取反，减少直流分量
                    //    tempChunk[i] = (i % 2 == 0) ? epsilon : -epsilon;
                    //}
                    tempChunk = tempChunk.Select(x => x == 0 ? -0.00070269317413142F : x).ToArray();
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(tempChunk, 0, decodeChunk, i * _frameLength + _frameLength - len, len);
                    }
                }
                return decodeChunk;
            }
        }

        public void RemoveDecodedChunk(int shiftLength)
        {
            if (shiftLength <= 0)
            {
                return;
            }
            lock (obj)
            {
                int oRowLen = (int)Math.Floor((double)_onlineInputEntity.Speech.Length / _featureDim);
                if (shiftLength < oRowLen)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    float[]? featuresTemp = new float[(oRowLen - shiftLength) * _featureDim];
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(features, i * oRowLen + shiftLength, featuresTemp, i * (oRowLen - shiftLength), oRowLen - shiftLength);
                    }
                    _onlineInputEntity.Speech = featuresTemp;
                    _onlineInputEntity.SpeechLength = featuresTemp.Length;
                }
                else
                {
                    _onlineInputEntity.Speech = new float[0];
                    _onlineInputEntity.SpeechLength = 0;

                    // 尝试将未检测到时间戳端点的数据进行保留
                    //float[]? features = _onlineInputEntity.Speech;
                    //float[]? featuresTemp = new float[502 * _featureDim];
                    //for (int i = 0; i < _featureDim; i++)
                    //{
                    //    Array.Copy(features, i * oRowLen, featuresTemp, i * 502, 502);
                    //}
                    //_onlineInputEntity.Speech = featuresTemp;
                    //_onlineInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }

        public List<float[]> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> statesList = new List<float[]>();
            return statesList;
        }

        /// <summary>
        /// when is endpoint,determine whether it is completed
        /// </summary>
        /// <param name="isEndpoint"></param>
        /// <returns></returns>
        public bool IsFinished(bool isEndpoint = false)
        {
            int featureDim = _frontendConfEntity.n_mels;
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
                        if (oLen <= _frameLength * featureDim)
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
        ~OnlineStream()
        {
            Dispose(_disposed);
        }
    }
}
