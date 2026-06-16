// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.ASR.Model;

namespace ManySpeech.ASR
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
        private List<string>? _hotwords = new List<string>();

        private static object obj = new object();
        private int _offset = 0;
        private int _requiredCacheSize = 0;
        private List<float[]> _caches = new List<float[]>();
        private List<float[]> _states = new List<float[]>();
        internal OfflineStream(IOffline offlineProj)
        {
            _offlineInputEntity = new OfflineInputEntity();
            //_wavFrontend = new WavFrontend(
            //    frontendConf: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.use_wavfrontend ? offlineProj.OfflineModel.ConfEntity.frontend_conf : null,
            //    mvnFilePath: offlineProj.OfflineModel.MvnFilePath,
            //    sampleRate: offlineProj.SampleRate,
            //    speechLength: offlineProj.SpeechLength,
            //    isResizeAudioDuration: offlineProj.IsResizeAudioDuration,
            //    isPaddingSpeech: offlineProj.IsPaddingSpeech,
            //    isSampleScalingRequired: offlineProj.IsSampleScalingRequired
            //    );
            _wavFrontend = new WavFrontend(
                frontendConf: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.use_wavfrontend ? offlineProj.OfflineModel.ConfEntity.frontend_conf : null,
                mvnFilePath: offlineProj.OfflineModel.MvnFilePath,
                sampleRate: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.fs,
                speechLength: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.speech_length,
                isResizeAudioDuration: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.is_resize_audio_duration,
                isPaddingSpeech: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.is_padding_speech,
                isSampleScalingRequired: offlineProj.OfflineModel.ConfEntity.preprocessor_conf.is_sample_scaling_required
                );
            _tokenIds = offlineProj.GetDecoderInitTokenIds();
            for (int i = 0; i < _tokenIds.Count; i++)
            {
                _timestamps.Add(new int[2]);
            }
        }

        public OfflineInputEntity OfflineInputEntity { get => _offlineInputEntity; set => _offlineInputEntity = value; }
        public List<int> TokenIds { get => _tokenIds; set => _tokenIds = value; }
        public List<string>? Tokens { get => _tokens; set => _tokens = value; }
        public List<int[]> Timestamps { get => _timestamps; set => _timestamps = value; }
        public List<string>? Hotwords { get => _hotwords; set => _hotwords = value; }
        public string? Region { get => _region; set => _region = value; }
        public string? Language { get => _language; set => _language = value; }
        public int Offset { get => _offset; set => _offset = value; }
        public int RequiredCacheSize { get => _requiredCacheSize; set => _requiredCacheSize = value; }
        public List<float[]> Caches { get => _caches; set => _caches = value; }
        public List<float[]> States { get => _states; set => _states = value; }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                float[] features = _wavFrontend.GetFeatures(samples);
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
                OfflineInputEntity.Language = Language;
                OfflineInputEntity.Region = Region;
            }
        }
        public OfflineInputEntity GetDecodeChunk()
        {
            lock (obj)
            {
                if (OfflineInputEntity.Speech != null && OfflineInputEntity.SpeechLength > 0)
                {
                    OfflineInputEntity.Hotwords = Hotwords;
                    OfflineInputEntity.Language = Language;
                    OfflineInputEntity.Region = Region;
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
