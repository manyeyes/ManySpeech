// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad
{
    public class OfflineStream:IDisposable
    {
        bool _disposed;

        private ModelInputEntity _modelInputEntity;
        private List<float> _speechProbList = new List<float>();

        // model config
        private int _window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
        private int _sample_rate;  //Assign when init support 16000 or 8000      
        private int _sr_per_ms;   // Assign when init, support 8 or 16
        private float _threshold;
        private int _min_silence_samples; // sr_per_ms * #ms
        private float _min_silence_samples_at_max_speech; // sr_per_ms * #98
        private int _min_speech_samples; // sr_per_ms * #ms
        private float _max_speech_samples;
        private int _speech_pad_samples; // usually a 
        private int _audio_length_samples;
        // model states
        private bool _triggered = false;
        private uint _temp_end = 0;
        private uint _current_sample = 0;
        // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes  
        private int _prev_end;
        private int _next_start = 0;

        //Output timestamp
        private List<SegmentEntity> _segments;
        private SegmentEntity _current_segment;
        //private List<int> _timestampsList = new List<int>();
        //private int[] _timestamps;
        //
        private float[]? _waveform = null;
        private List<float[]> _waveforms = new List<float[]>();
        private int _lastWaveFormsNum = 0;
        //Onnx Model Output states
        private List<float[]> _states = new List<float[]>();
        //
        private int _chunkLength = 512;
        private int _shiftLength = 512;
        private ModelCustomMetadata? _customMetadata;
        //obj of lock
        private static object obj = new object();
        //__DEBUG_SPEECH_PROB___
        private bool _isDebug = false;
        internal OfflineStream(IVadProj? vadProj)
        {
            if (vadProj != null)
            {
                _states = vadProj.GetModelInitStates();
                _customMetadata = vadProj.CustomMetadata;
                _sample_rate = vadProj.SampleRate;
                _isDebug = vadProj.IsDebug;
            }
            _modelInputEntity = new ModelInputEntity();
            _segments = new List<SegmentEntity>();
            _current_segment = new SegmentEntity();
            InitStream();
            _chunkLength = (int)_window_size_samples;
            _shiftLength = (int)_window_size_samples;
        }

        public ModelInputEntity ModelInputEntity { get => _modelInputEntity; set => _modelInputEntity = value; }
        public List<float[]> States { get => _states; set => _states = value; }
        public List<SegmentEntity> Segments { get => _segments; set => _segments = value; }
        public SegmentEntity Current_segment { get => _current_segment; set => _current_segment = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public List<float[]> Waveforms { get => _waveforms; set => _waveforms = value; }
        public float[]? Waveform { get => _waveform; set => _waveform = value; }
        public int LastWaveFormsNum { get => _lastWaveFormsNum; set => _lastWaveFormsNum = value; }

        //public List<int> TimestampsList { get => _timestampsList; set => _timestampsList = value; }
        //public int[] Timestamps { get => _timestamps; set => _timestamps = value; }

        public void InitStream()
        {
            _threshold = _customMetadata.threshold;
            _sr_per_ms = _sample_rate / 1000;

            _window_size_samples = _sr_per_ms * _customMetadata.windows_frame_size;

            _min_speech_samples = _sr_per_ms * _customMetadata.min_speech_duration_ms;
            _speech_pad_samples = _sr_per_ms * _customMetadata.speech_pad_ms;

            _max_speech_samples = (
                _sample_rate * _customMetadata.max_speech_duration_s
                - _window_size_samples
                - 2 * _speech_pad_samples
                );

            _min_silence_samples = _sr_per_ms * _customMetadata.min_silence_duration_ms;
            _min_silence_samples_at_max_speech = _sr_per_ms * 98;
        }

        public void PushIndex(float speech_prob)
        {
            _speechProbList.Add(speech_prob);
        }
        /// <summary>
        /// refer to: https://github.com/snakers4/silero-vad/blob/master/examples/csharp/SileroVadDetector.cs#L72
        /// </summary>
        public void CalculateProb()
        {
            List<float> speechProbList = _speechProbList;
            List<SegmentEntity> result = new List<SegmentEntity>();
            bool triggered = false;
            int tempEnd = 0, prevEnd = 0, nextStart = 0;
            SegmentEntity segment = new SegmentEntity();

            for (int i = 0; i < speechProbList.Count; i++)
            {
                float speechProb = speechProbList[i];
                if (speechProb >= _threshold && (tempEnd != 0))
                {
                    tempEnd = 0;
                    if (nextStart < prevEnd)
                    {
                        nextStart = _window_size_samples * i;
                    }
                }

                if (speechProb >= _threshold && !triggered)
                {
                    triggered = true;
                    segment.Start = _window_size_samples * i;
                    continue;
                }

                if (triggered && (_window_size_samples * i) - segment.Start > _max_speech_samples)
                {
                    if (prevEnd != 0)
                    {
                        segment.End = prevEnd;
                        result.Add(segment);
                        segment = new SegmentEntity();
                        if (nextStart < prevEnd)
                        {
                            triggered = false;
                        }
                        else
                        {
                            segment.Start = nextStart;
                        }

                        prevEnd = 0;
                        nextStart = 0;
                        tempEnd = 0;
                    }
                    else
                    {
                        segment.End = _window_size_samples * i;
                        result.Add(segment);
                        segment = new SegmentEntity();
                        prevEnd = 0;
                        nextStart = 0;
                        tempEnd = 0;
                        triggered = false;
                        continue;
                    }
                }

                if (speechProb < _threshold && triggered)
                {
                    if (tempEnd == 0)
                    {
                        tempEnd = _window_size_samples * i;
                    }

                    if (((_window_size_samples * i) - tempEnd) > _min_silence_samples_at_max_speech)
                    {
                        prevEnd = tempEnd;
                    }

                    if ((_window_size_samples * i) - tempEnd < _min_silence_samples)
                    {
                        continue;
                    }
                    else
                    {
                        segment.End = tempEnd;
                        if ((segment.End - segment.Start) > _min_speech_samples)
                        {
                            result.Add(segment);
                        }

                        segment = new SegmentEntity();
                        prevEnd = 0;
                        nextStart = 0;
                        tempEnd = 0;
                        triggered = false;
                        continue;
                    }
                }
            }

            if (segment.Start != null && (_audio_length_samples - segment.Start) > _min_speech_samples)
            {
                segment.End = _audio_length_samples;
                result.Add(segment);
            }

            for (int i = 0; i < result.Count; i++)
            {
                SegmentEntity item = result[i];
                if (i == 0)
                {
                    item.Start = (int)Math.Max(0, item.Start - _speech_pad_samples);
                }

                if (i != result.Count - 1)
                {
                    SegmentEntity nextItem = result[i + 1];
                    int silenceDuration = nextItem.Start - item.End;
                    if (silenceDuration < 2 * _speech_pad_samples)
                    {
                        item.End = item.End + (silenceDuration / 2);
                        nextItem.Start = Math.Max(0, nextItem.Start - (silenceDuration / 2));
                    }
                    else
                    {
                        item.End = (int)Math.Min(_audio_length_samples, item.End + _speech_pad_samples);
                        nextItem.Start = (int)Math.Max(0, nextItem.Start - _speech_pad_samples);
                    }
                }
                else
                {
                    item.End = (int)Math.Min(_audio_length_samples, item.End + _speech_pad_samples);
                }
            }
            _segments = MergeListAndCalculateSecond(result, _sample_rate);
        }
        /// <summary>
        /// refer to: https://github.com/snakers4/silero-vad/blob/master/examples/csharp/SileroVadDetector.cs#L203C39-L203C66
        /// </summary>
        private List<SegmentEntity> MergeListAndCalculateSecond(List<SegmentEntity> original, int samplingRate)
        {
            List<SegmentEntity> result = new List<SegmentEntity>();
            if (original == null || original.Count == 0)
            {
                return result;
            }

            int left = original[0].Start;
            int right = original[0].End;
            if (original.Count > 1)
            {
                original.Sort((a, b) => a.Start.CompareTo(b.Start));
                for (int i = 1; i < original.Count; i++)
                {
                    SegmentEntity segment = original[i];

                    if (segment.Start > right)
                    {
                        result.Add(new SegmentEntity(left, right));
                        left = segment.Start;
                        right = segment.End;
                    }
                    else
                    {
                        right = Math.Max(right, segment.End);
                    }
                }
                result.Add(new SegmentEntity(left, right));
            }
            else
            {
                result.Add(new SegmentEntity(left, right));
            }

            return result;
        }

        public static Int16 Float32ToInt16(float sample)
        {
            if (sample < -0.999999f)
            {
                return Int16.MinValue;
            }
            else if (sample > 0.999999f)
            {
                return Int16.MaxValue;
            }
            else
            {
                if (sample < 0)
                {
                    return (Int16)(Math.Floor(sample * 32767.0f));
                }
                else
                {
                    return (Int16)(Math.Ceiling(sample * 32767.0f));
                }
            }
        }

        public float[] GetSamples(float[] samples)
        {
            samples = samples.Select((float x) => x).ToArray();
            return samples;
        }

        public void AddSamples(float[] samples)
        {
            lock (obj)
            {
                int oLen = 0;
                if (ModelInputEntity.SpeechLength > 0)
                {
                    oLen = ModelInputEntity.SpeechLength;
                }
                float[] features = GetSamples(samples);
                if (features.Length > 0)
                {
                    float[]? featuresTemp = new float[oLen + features.Length];
                    if (ModelInputEntity.SpeechLength > 0)
                    {
                        Array.Copy(ModelInputEntity.Speech, 0, featuresTemp, 0, ModelInputEntity.SpeechLength);
                    }
                    Array.Copy(features, 0, featuresTemp, ModelInputEntity.SpeechLength, features.Length);
                    ModelInputEntity.Speech = featuresTemp;
                    ModelInputEntity.SpeechLength = featuresTemp.Length;
                }
                //cache full samples
                if (features.Length > 0)
                {
                    float[]? featuresTemp = _waveform == null ? new float[features.Length] : new float[_waveform.Length + features.Length];
                    if (_waveform != null)
                    {
                        Array.Copy(_waveform, 0, featuresTemp, 0, _waveform.Length);
                        Array.Copy(features, 0, featuresTemp, _waveform.Length, features.Length);
                        _waveform = featuresTemp;
                    }
                    else
                    {
                        _waveform = new float[features.Length];
                        Array.Copy(features, 0, _waveform, 0, _waveform.Length);
                    }
                    _audio_length_samples = _waveform.Length;
                }
            }
        }

        public float[]? GetDecodeChunk(int chunkLength)
        {
            int featureDim = 1;
            lock (obj)
            {
                float[]? padChunk = new float[chunkLength * featureDim];
                if (chunkLength * featureDim <= ModelInputEntity.SpeechLength)
                {
                    float[]? features = ModelInputEntity.Speech;
                    Array.Copy(features, 0, padChunk, 0, padChunk.Length);
                    return padChunk;
                }
                else
                {
                    return null;
                }
            }
        }

        public void RemoveChunk(int shiftLength)
        {
            lock (obj)
            {
                int featureDim = 1;
                if (shiftLength * featureDim <= ModelInputEntity.SpeechLength)
                {
                    float[]? features = ModelInputEntity.Speech;
                    float[]? featuresTemp = new float[features.Length - shiftLength * featureDim];
                    Array.Copy(features, shiftLength * featureDim, featuresTemp, 0, featuresTemp.Length);
                    ModelInputEntity.Speech = featuresTemp;
                    ModelInputEntity.SpeechLength = featuresTemp.Length;
                }
            }
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_segments != null)
                    {
                        _segments=null;
                    }
                    if (_current_segment != null)
                    {
                        _current_segment = null;
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
