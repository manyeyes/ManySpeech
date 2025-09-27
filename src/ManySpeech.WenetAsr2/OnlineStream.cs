// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.WenetAsr2.Model;

namespace ManySpeech.WenetAsr2
{
    public class OnlineStream
    {
        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private OnlineInputEntity _onlineInputEntity;
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
        private float[] _cacheFeats = null;
        private float[] _cacheInput = null;
        private float[] _cachelfrSplice = null;
        private int _frame_sample_length;
        private int _frame_shift_sample_length;
        private int _lfr_m = 11;
        private float[] _cacheSamples = null;
        private int _offset = 0;
        private int _required_cache_size = 0;
        public OnlineStream(OnlineModel onlineModel)
        {
            if (onlineModel != null)
            {
                _chunkLength = onlineModel.ChunkLength;
                _shiftLength = onlineModel.ShiftLength;
                _featureDim = onlineModel.FeatureDim;
                _sampleRate = onlineModel.SampleRate;
                _customMetadata = onlineModel.CustomMetadata;
                _required_cache_size = onlineModel.Required_cache_size;
                if (_required_cache_size > 0)
                {
                    _offset = _required_cache_size;
                }
            }
            _onlineInputEntity = new OnlineInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;

            _wavFrontend = new WavFrontend(_frontendConfEntity);
            _hyp = new Int64[] { _blank_id, _blank_id };
            _states = GetEncoderInitStates();
            _cacheFeats = InitCacheFeats();
            _cacheSamples = new float[160 * _chunkLength];
            _tokens = new List<Int64> { _blank_id, _blank_id };

            _frame_sample_length = 25 * 16000 / 1000;
            _frame_shift_sample_length = 10 * 16000 / 1000;
            _cacheInput = new float[320];
            _cacheInput = _cacheInput.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
        }

        public OnlineInputEntity OnlineInputEntity { get => _onlineInputEntity; set => _onlineInputEntity = value; }
        public long[] Hyp { get => _hyp; set => _hyp = value; }
        public List<Int64> Tokens { get => _tokens; set => _tokens = value; }
        public List<int> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }

        private int ComputeFrameNum(int samplesLength)
        {
            int frameNum = (samplesLength - _frame_sample_length) / _frame_shift_sample_length + 1;
            if (frameNum < 1 || samplesLength < _frame_sample_length)
            {
                frameNum = 0;
            }
            return frameNum;
        }

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
                int chunkSamplesLength = 160 * _chunkLength;
                if (cacheSamplesLength > chunkSamplesLength)
                {
                    //get head
                    float[] _samples = new float[chunkSamplesLength];
                    Array.Copy(_cacheSamples, 0, _samples, 0, _samples.Length);
                    InputSpeech(_samples);
                    //remove head
                    float[] _cacheSamplesTemp = new float[cacheSamplesLength - chunkSamplesLength];
                    Array.Copy(_cacheSamples, chunkSamplesLength, _cacheSamplesTemp, 0, _cacheSamplesTemp.Length);
                    _cacheSamples = _cacheSamplesTemp;
                }
            }
        }

        public void InputSpeech(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    oLen = OnlineInputEntity.SpeechLength;
                }
                float[] inputs = new float[samples.Length];
                if (_cacheInput != null)
                {
                    inputs = new float[_cacheInput.Length + samples.Length];
                    Array.Copy(_cacheInput, 0, inputs, 0, _cacheInput.Length);
                    Array.Copy(samples, 0, inputs, _cacheInput.Length, samples.Length);
                }
                else
                {
                    Array.Copy(samples, 0, inputs, 0, samples.Length);
                }
                int frameNum = ComputeFrameNum(inputs.Length);
                // compute fbank
                int waveformLength = inputs.Length;// (frameNum - 1) * _frame_shift_sample_length + _frame_sample_length;
                float[] waveform = new float[waveformLength];
                Array.Copy(inputs, 0, waveform, 0, waveform.Length);
                float[] features = _wavFrontend.GetFbank(waveform);

                if (_cacheInput == null)
                {
                    int repeatNum = (_lfr_m - 1) / 2 - 1;
                    int featureDim = _frontendConfEntity.n_mels;
                    float[] firstFbank = new float[featureDim];
                    //Array.Copy(features, 0, firstFbank, 0, firstFbank.Length);
                    float[] features_temp = new float[firstFbank.Length * repeatNum + features.Length];
                    for (int i = 0; i < repeatNum; i++)
                    {
                        Array.Copy(firstFbank, 0, features_temp, i * firstFbank.Length, firstFbank.Length);
                    }
                    Array.Copy(features, 0, features_temp, firstFbank.Length * repeatNum, features.Length);
                    features = features_temp;
                }

                // compute cacheInput
                int cacheInputLength = inputs.Length - frameNum * _frame_shift_sample_length;
                _cacheInput = new float[cacheInputLength];
                Array.Copy(inputs, inputs.Length - cacheInputLength, _cacheInput, 0, cacheInputLength);

                //_cachelfrSplice
                _cachelfrSplice = new float[80];
                Array.Copy(features, features.Length - 80, _cachelfrSplice, 0, _cachelfrSplice.Length);

                float[]? featuresTemp = new float[oLen + features.Length];
                if (OnlineInputEntity.SpeechLength > 0)
                {
                    Array.Copy(_onlineInputEntity.Speech, 0, featuresTemp, 0, _onlineInputEntity.SpeechLength);
                }
                Array.Copy(features, 0, featuresTemp, OnlineInputEntity.SpeechLength, features.Length);
                OnlineInputEntity.Speech = featuresTemp;
                OnlineInputEntity.SpeechLength = featuresTemp.Length;
            }
        }

        // Note: chunk_length is in frames before subsampling
        public float[]? GetDecodeChunk(int chunkLength)
        {
            int featureDim = _frontendConfEntity.n_mels;
            lock (obj)
            {
                float[]? decodeChunk = null;
                //use non-streaming asr,get all chunks
                if (chunkLength <= 0)
                {
                    chunkLength = _onlineInputEntity.SpeechLength/ featureDim;
                }
                if (chunkLength < 67)
                {
                    return decodeChunk;
                }
                //use non-streaming asr,get all chunks

                if (chunkLength * featureDim <= _onlineInputEntity.SpeechLength)
                {

                    float[] padChunk = new float[chunkLength * featureDim];
                    //Array.Copy(_cacheFeats, 0, padChunk, 0, _cacheFeats.Length);
                    float[]? features = _onlineInputEntity.Speech;
                    //padChunk = new float[chunkLengthPad * _featureDim];
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);                    
                    decodeChunk = new float[_cacheFeats.Length + padChunk.Length];
                    Array.Copy(_cacheFeats, 0, decodeChunk, 0, _cacheFeats.Length);
                    Array.Copy(padChunk, 0, decodeChunk, _cacheFeats.Length, padChunk.Length);
                }
                return decodeChunk;
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = _frontendConfEntity.n_mels;
                if (shiftLength * featureDim <= _onlineInputEntity.SpeechLength)
                {
                    float[]? features = _onlineInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    _onlineInputEntity.Speech = featuresTemp;
                    _onlineInputEntity.SpeechLength = featuresTemp.Length;
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

        private float[] InitCacheFeats(int batchSize = 1)
        {

            int cached_feature_size = 0;//1 + _right_context - _subsampling_rate;//TODO temp test
            float[] cacheFeats = new float[batchSize * cached_feature_size * 80];
            return cacheFeats;
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

    }
}
