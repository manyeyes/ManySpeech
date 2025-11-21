// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.WhisperAsr.Model;

namespace ManySpeech.WhisperAsr
{
    public class TranscribeStream : IDisposable
    {
        private bool _disposed;

        private FrontendConfEntity _frontendConfEntity;
        private WavFrontend _wavFrontend;
        private TranscribeInputEntity _transcribeInputEntity;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 30;
        private int _frameLength = 3000; // 3000 frames
        private int _hopLength = 160;
        private int _shiftLength = 0;
        private float _suppressSample;

        private CustomMetadata? _customMetadata;
        private List<int> _tokens = new List<int>();
        private string? _language;
        private List<int[]> _timestamps = new List<int[]>();
        private List<float[]> _states = new List<float[]>();
        private List<SegmentEntity> _allSegments = new List<SegmentEntity>();
        private static object obj = new object();
        private int _offset = 0;
        private int _realSampleLen = 0;
        public TranscribeStream(TranscribeModel? transcribeModel)
        {
            if (transcribeModel != null)
            {
                _chunkLength = transcribeModel.ChunkLength;
                _frameLength = transcribeModel.FrameLength;
                _shiftLength = transcribeModel.ShiftLength;
                _hopLength = transcribeModel.HopLength;
                _shiftLength = transcribeModel.ShiftLength;
                _featureDim = transcribeModel.FeatureDim;
                _sampleRate = transcribeModel.SampleRate;
                _suppressSample = transcribeModel.SuppressSample;
                _customMetadata = transcribeModel.CustomMetadata;
            }

            _transcribeInputEntity = new TranscribeInputEntity();
            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = _sampleRate;
            _frontendConfEntity.n_mels = _featureDim;

            _wavFrontend = new WavFrontend(_frontendConfEntity);

        }

        public TranscribeInputEntity TranscribeInputEntity { get => _transcribeInputEntity; set => _transcribeInputEntity = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public List<int> Tokens { get => _tokens; set => _tokens = value; }
        public int RealSampleLen { get => _realSampleLen; set => _realSampleLen = value; }
        public string? Language { get => _language; set => _language = value; }
        public List<SegmentEntity> AllSegments { get => _allSegments; set => _allSegments = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int chunkSamplesLength = _hopLength * _frameLength;
                int numStep = (int)Math.Ceiling((double)samples.Length / chunkSamplesLength);
                for (int i = 0; i < numStep; i++)
                {
                    float[] chunkSamples = new float[chunkSamplesLength];
                    int len = chunkSamplesLength;
                    if (chunkSamplesLength > samples.Length)
                    {
                        len = samples.Length;
                    }
                    else
                    {
                        if (chunkSamplesLength * (i + 1) > samples.Length)
                        {
                            len = samples.Length - chunkSamplesLength * i;
                        }
                    }
                    Array.Copy(samples, i * chunkSamplesLength, chunkSamples, 0, len);
                    InputSpeech(chunkSamples);
                }
            }
        }
        public void InputSpeech(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                int oRowLen = 0;
                if (TranscribeInputEntity.SpeechLength > 0)
                {
                    oLen = TranscribeInputEntity.SpeechLength;
                    oRowLen = TranscribeInputEntity.SpeechLength / _featureDim;
                }
                float[] features = _wavFrontend.GetFeatures(samples);
                int featuresRowLen = features.Length / _featureDim;

                float[]? featuresTemp = new float[oLen + features.Length];//new matrix
                int featuresTempRowLen = featuresTemp.Length / _featureDim;
                if (TranscribeInputEntity.SpeechLength > 0)
                {
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(TranscribeInputEntity.Speech, i * oRowLen, featuresTemp, i * featuresTempRowLen, oRowLen);
                    }
                }
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(features, i * featuresRowLen, featuresTemp, i * featuresTempRowLen + oRowLen, featuresRowLen);
                }
                TranscribeInputEntity.Speech = featuresTemp;
                TranscribeInputEntity.SpeechLength = featuresTemp.Length;
            }
        }
        public float[]? GetAllDecodeChunk()
        {
            lock (obj)
            {
                float[]? decodeChunk = null;
                decodeChunk = TranscribeInputEntity.Speech;
                return decodeChunk;
            }
        }
        public void RemoveAllDecodedChunk()
        {
            lock (obj)
            {
                if (_tokens.Count > 2)
                {
                    TranscribeInputEntity.Speech = null;
                    TranscribeInputEntity.SpeechLength = 0;
                }
            }
        }
        public float[]? GetOneDecodeChunk()
        {
            lock (obj)
            {
                float[]? decodeChunk = new float[_frameLength * _featureDim];
                int oRowLen = _transcribeInputEntity.SpeechLength / _featureDim;
                for (int i = 0; i < _featureDim; i++)
                {
                    Array.Copy(TranscribeInputEntity.Speech, i * oRowLen, decodeChunk, i * _frameLength, _frameLength);
                }
                return decodeChunk;
            }
        }
        public void RemoveDecodedChunk(int shiftLength)
        {
            lock (obj)
            {
                if (shiftLength * _featureDim <= _transcribeInputEntity.SpeechLength)
                {
                    float[]? features = _transcribeInputEntity.Speech;
                    int oRowLen = _transcribeInputEntity.SpeechLength / _featureDim;
                    float[]? featuresTemp = new float[(oRowLen - shiftLength) * _featureDim];
                    for (int i = 0; i < _featureDim; i++)
                    {
                        Array.Copy(features, i * oRowLen + shiftLength, featuresTemp, i * (oRowLen - shiftLength), oRowLen - shiftLength);
                    }
                    _transcribeInputEntity.Speech = featuresTemp;
                    _transcribeInputEntity.SpeechLength = featuresTemp.Length;
                }
                //else
                //{
                //    _transcribeInputEntity.Speech = null;
                //    _transcribeInputEntity.SpeechLength = 0;
                //}
            }
        }

        public List<Int64[]> GetDecoderInitTokens(int nAudio, int tokenizerSot)
        {
            List<Int64[]> tokens = new List<Int64[]>();
            Int64[] longItem = new Int64[] { tokenizerSot };//50257
            for (int i = 0; i < Math.Min(nAudio, tokens.Count); i++)
            {
                tokens.Add(longItem);
            }
            return tokens;
        }

        /// <summary>
        /// 
        /// The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        /// tensors calculated for the previous positions.This method returns a dictionary that stores
        /// all caches, and the necessary hooks for the key and value projection modules that save the
        /// intermediate tensors to be reused during later calculations.
        /// Returns
        /// -------
        /// cache : Dict[nn.Module, torch.Tensor]
        ///     A dictionary object mapping the key/value projection modules to its cache
        /// hooks : List[RemovableHandle]
        ///     List of PyTorch RemovableHandle objects to stop the hooks to be called
        /// </summary>
        public void InstallKvCacheHooks()
        {

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
                    if (_transcribeInputEntity != null)
                    {
                        _transcribeInputEntity = null;
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
        ~TranscribeStream()
        {
            Dispose(_disposed);
        }
    }
}
