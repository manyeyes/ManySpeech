// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad
{
    public class OnlineStream:IDisposable
    {
        bool _disposed;

        private ModelInputEntity _modelInputEntity;

        // model config
        private Int64 _window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
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
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private ModelCustomMetadata? _customMetadata;
        //obj of lock
        private static object obj = new object();
        //__DEBUG_SPEECH_PROB___
        private bool _isDebug = false;
        internal OnlineStream(IVadProj? vadProj)
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
            //Current_segment=new SegmentEntity();
            // Push forward sample index
            _current_sample += (uint)_window_size_samples;

            // Reset temp_end when > threshold 
            if (speech_prob >= _threshold)
            {
                if (_isDebug)
                {
                    float speech = _current_sample - _window_size_samples; // minus window_size_samples to get precise start time point.
                    Console.WriteLine(string.Format("start: {0} s ({1}) {2}\n", 1.0 * speech / _sample_rate, speech_prob, _current_sample - _window_size_samples));
                }
                if (_temp_end != 0)
                {
                    _temp_end = 0;
                    if (_next_start < _prev_end)
                    {
                        _next_start = (int)(_current_sample - _window_size_samples);
                    }
                }
                if (!_triggered)
                {
                    _triggered = true;
                    Current_segment.Start = (int)(_current_sample - _window_size_samples);
                }
                return;
            }

            if (
                _triggered &&
                (_current_sample - _window_size_samples - Current_segment.Start) > _max_speech_samples
                )
            {
                if (_prev_end > 0)
                {
                    Current_segment.End = _prev_end;
                    Segments.Add(Current_segment);
                    Current_segment = new SegmentEntity();

                    // previously reached silence(< neg_thres) and is still not speech(< thres)
                    if (_next_start < _prev_end)
                    {
                        _triggered = false;
                    }
                    else
                    {
                        Current_segment.Start = _next_start;
                    }
                    _prev_end = 0;
                    _next_start = 0;
                    _temp_end = 0;

                }
                else
                {
                    Current_segment.End = (int)(_current_sample - _window_size_samples);
                    Segments.Add(Current_segment);
                    Current_segment = new SegmentEntity();
                    _prev_end = 0;
                    _next_start = 0;
                    _temp_end = 0;
                    _triggered = false;
                }
                return;
            }
            if ((speech_prob >= (_threshold - 0.15)) && (speech_prob < _threshold))
            {
                if (_triggered)
                {
                    if (_isDebug)
                    {
                        float speech = _current_sample - _window_size_samples; // minus window_size_samples to get precise start time point.
                        Console.WriteLine(string.Format("speeking: {0} s ({1}) {2}\n", 1.0 * speech / _sample_rate, speech_prob, _current_sample - _window_size_samples));
                    }
                }
                else
                {
                    if (_isDebug)
                    {
                        float speech = _current_sample - _window_size_samples; // minus window_size_samples to get precise start time point.
                        Console.WriteLine(string.Format("silence: {0} s ({1}) {2}\n", 1.0 * speech / _sample_rate, speech_prob, _current_sample - _window_size_samples));
                    }
                }
                return;
            }


            // 4) End 
            if (speech_prob < _threshold - 0.15)
            {
                if (_isDebug)
                {
                    float speech = _current_sample - _window_size_samples - _speech_pad_samples; // minus window_size_samples to get precise start time point.
                    Console.WriteLine(string.Format("end: {0} s ({1}) {2}\n", 1.0 * speech / _sample_rate, speech_prob, _current_sample - _window_size_samples));
                }
                if (_triggered)
                {
                    if (_temp_end == 0)
                    {
                        _temp_end = (uint)(_current_sample - _window_size_samples);
                    }
                    if (_current_sample - _window_size_samples - _temp_end > _min_silence_samples_at_max_speech)
                    {
                        _prev_end = (int)_temp_end;
                    }
                    // a. silence < min_slience_samples, continue speaking 
                    if ((_current_sample - _window_size_samples - _temp_end) < _min_silence_samples)
                    {

                    }
                    // b. silence >= min_slience_samples, end speaking
                    else
                    {
                        Current_segment.End = (int)_temp_end;
                        if (Current_segment.End - Current_segment.Start > _min_speech_samples)
                        {
                            Segments.Add(Current_segment);
                            Current_segment = new SegmentEntity();
                            _prev_end = 0;
                            _next_start = 0;
                            _temp_end = 0;
                            _triggered = false;
                        }
                    }
                }
                else
                {
                    // may first windows see end state.
                }
                return;
            }
            if (Current_segment.Start != null && _audio_length_samples - Current_segment.Start > _min_speech_samples)
            {
                Current_segment.End = _audio_length_samples;
            }
        }

        public void PushIndexEnd()
        {
            if (Current_segment.Start > 0)
            {
                Current_segment.End = _audio_length_samples;
                Segments.Add(Current_segment);
                Current_segment = new SegmentEntity();
                _prev_end = 0;
                _next_start = 0;
                _temp_end = 0;
                _triggered = false;
            }
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

        /// <summary>
        /// when is endpoint,determine whether it is completed
        /// </summary>
        /// <param name="isEndpoint"></param>
        /// <returns></returns>
        public bool IsFinished(bool isEndpoint = false)
        {
            int featureDim = 1;
            if (isEndpoint)
            {
                int oLen = 0;
                if (ModelInputEntity.SpeechLength > 0)
                {
                    oLen = ModelInputEntity.SpeechLength;
                }
                if (oLen > 0)
                {
                    var avg = ModelInputEntity.Speech.Average();
                    int num = ModelInputEntity.Speech.Where(x => x != avg).ToArray().Length;
                    //int num = ModelInputEntity.Speech.Where(x => (float)Math.Round((double)x, 5) != (float)Math.Round((double)avg, 5)).ToArray().Length;
                    if (num == 0)
                    {
                        PushIndexEnd();
                        return true;
                    }
                    else
                    {
                        if (oLen <= _chunkLength * featureDim)
                        {
                            AddSamples(new float[1024]);
                        }
                        return false;
                    }

                }
                else
                {
                    PushIndexEnd();
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
                    if (_segments != null)
                    {
                        _segments = null;
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
        ~OnlineStream()
        {
            Dispose(_disposed);
        }
    }
}
