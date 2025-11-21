using Microsoft.ML.OnnxRuntime;
using ManySpeech.MoonshineAsr.Model;
using System.Reflection;

namespace ManySpeech.MoonshineAsr
{
    internal class AsrModel
    {
        private InferenceSession _preprocessSession;
        private InferenceSession _encodeSession;
        private InferenceSession _cachedDecodeSession;
        private InferenceSession _uncachedDecodeSession;

        private CustomMetadata? _customMetadata;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _sos_eos_id = 0;

        private int _featureDim = 1;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;

        public AsrModel(string preprocessFilePath, string encodeFilePath, string cachedDecodeFilePath, string uncachedDecodeFilePath, string configFilePath = "", int threadsNum = 2)
        {
            _preprocessSession = initModel(preprocessFilePath, threadsNum);
            _encodeSession = initModel(encodeFilePath, threadsNum);
            _cachedDecodeSession = initModel(cachedDecodeFilePath, threadsNum);
            _uncachedDecodeSession = initModel(uncachedDecodeFilePath, threadsNum);

            _customMetadata = LoadConf(configFilePath);

            _chunkLength = _sampleRate * 10;
            _shiftLength = _sampleRate * 10;
            _featureDim = 1;
        }


        public CustomMetadata? CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public InferenceSession PreprocessSession { get => _preprocessSession; set => _preprocessSession = value; }
        public InferenceSession EncodeSession { get => _encodeSession; set => _encodeSession = value; }
        public InferenceSession CachedDecodeSession { get => _cachedDecodeSession; set => _cachedDecodeSession = value; }
        public InferenceSession UncachedDecodeSession { get => _uncachedDecodeSession; set => _uncachedDecodeSession = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2)
        {
            if (string.IsNullOrEmpty(modelFilePath) || !File.Exists(modelFilePath))
            {
                return null;
            }
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            //options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; // 启用所有图优化
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
            var assembly = typeof(AsrModel).Assembly;
            var stream = assembly.GetManifestResourceStream(resourceName) ??
                         throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");

            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            stream.Seek(0, SeekOrigin.Begin);
            stream.Close();
            stream.Dispose();

            return bytes;
        }

        private CustomMetadata? LoadConf(string configFilePath)
        {
            CustomMetadata? confJsonEntity = new CustomMetadata();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confJsonEntity = Utils.PreloadHelper.ReadJson<CustomMetadata>(configFilePath);
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath); // To compile for AOT
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<CustomMetadata>(configFilePath);
                }
            }
            return confJsonEntity;
        }

        private CustomMetadata? LoadJsonConf(string configFilePath)
        {
            if (string.IsNullOrWhiteSpace(configFilePath))
            {
                return null;
            }
            CustomMetadata? confJsonEntity = Utils.PreloadHelper.ReadJson<CustomMetadata>(configFilePath);
            return confJsonEntity;
        }
        private CustomMetadata? LoadYamlConf(string configFilePath)
        {
            if (string.IsNullOrWhiteSpace(configFilePath))
            {
                return null;
            }
            CustomMetadata? confJsonEntity = Utils.PreloadHelper.ReadYaml<CustomMetadata>(configFilePath);
            return confJsonEntity;
        }
    }
}
