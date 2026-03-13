// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.OmniAsr.Model;
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

namespace ManySpeech.OmniAsr
{
    /// <summary>
    /// offline model
    /// </summary>
    internal class OfflineModel
    {
        private InferenceSession _modelSession;

        private int _featureDim = 1;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;

        private ConfEntity? _confEntity;

        public OfflineModel(string modelFilePath, string configFilePath = "", int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);

            _sampleRate = _confEntity.sampleRate;

        }

        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public ConfEntity? ConfEntity { get => _confEntity; set => _confEntity = value; }

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
            //var assembly = Assembly.GetExecutingAssembly();
            var assembly = typeof(OfflineModel).Assembly;
            var stream = assembly.GetManifestResourceStream(resourceName) ??
                         throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");
            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            stream.Seek(0, SeekOrigin.Begin);
            stream.Close();
            stream.Dispose();

            return bytes;
        }

        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confJsonEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confJsonEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath);
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
                }
            }
            return confJsonEntity;
        }
    }
}
