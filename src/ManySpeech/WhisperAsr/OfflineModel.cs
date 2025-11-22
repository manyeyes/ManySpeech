// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.WhisperAsr.Model;
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

namespace ManySpeech.WhisperAsr
{
    public class OfflineModel : IDisposable
    {
        private bool _disposed;
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private CustomMetadata _customMetadata;
        //private ConfEntity? _confEntity;

        private bool _isMultilingual = false;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 30;
        private int _frameLength = 3000; // 3000 frames
        private int _hopLength = 160;
        private int _shiftLength = 0;
        private float _suppressSample = float.NaN;
        private int _required_cache_size = 0;

        public OfflineModel(string encoderFilePath, string decoderFilePath, ConfEntity confEntity, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);

            //_confEntity = LoadYamlConf(configFilePath);

            _featureDim = confEntity.model_dimensions.n_mels;

            _customMetadata = new CustomMetadata();
            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

            _isMultilingual = false;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public bool IsMultilingual { get => _isMultilingual; set => _isMultilingual = value; }
        //internal ConfEntity? ConfEntity { get => _confEntity; set => _confEntity = value; }
        public int HopLength { get => _hopLength; set => _hopLength = value; }
        public int FrameLength { get => _frameLength; set => _frameLength = value; }
        public float SuppressSample { get => _suppressSample; set => _suppressSample = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2)
        {
            if (string.IsNullOrEmpty(modelFilePath) || !File.Exists(modelFilePath))
            {
                return null;
            }
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            //options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            //options.AppendExecutionProvider_MKLDNN();
            //options.AppendExecutionProvider_ROCm(0);
            if (threadsNum > 0)
                options.InterOpNumThreads = threadsNum;
            else
                options.InterOpNumThreads = System.Environment.ProcessorCount;
            // 启用CPU内存计划
            options.EnableMemoryPattern = true;
            // 设置其他优化选项            
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            InferenceSession onnxSession = null;
            if (!string.IsNullOrEmpty(modelFilePath) && modelFilePath.IndexOf("/") < 0 && modelFilePath.IndexOf("\\") < 0)
            {
                byte[] model = ReadEmbeddedResourceAsBytes(modelFilePath);
                onnxSession = new InferenceSession(model, options);
            }
            else
            {
                onnxSession = new InferenceSession(modelFilePath, options);
            }
            return onnxSession;
        }

        private static byte[] ReadEmbeddedResourceAsBytes(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var stream = assembly.GetManifestResourceStream(resourceName) ??
                         throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            // 设置当前流的位置为流的开始 
            stream.Seek(0, SeekOrigin.Begin);
            stream.Close();
            stream.Dispose();

            return bytes;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_encoderSession != null)
                    {
                        _encoderSession.Dispose();
                    }
                    if (_decoderSession != null)
                    {
                        _decoderSession.Dispose();
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
        ~OfflineModel()
        {
            Dispose(_disposed);
        }
    }
}
