using Microsoft.ML.OnnxRuntime;
using ManySpeech.SileroVad.Model;
using ManySpeech.SileroVad.Utils;

namespace ManySpeech.SileroVad
{
    public class VadModel
    {
        private InferenceSession _modelSession;
        private ModelCustomMetadata? _customMetadata;
        public VadModel(string modelFilePath,string configFilePath="", float threshold = 0F, int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _customMetadata = LoadConf(configFilePath);
            if (_customMetadata == null)
            {
                _customMetadata = new ModelCustomMetadata();
            }
            if (threshold > 0F)
            {
                _customMetadata.threshold = threshold;
            }
            //var encoder_meta = _modelSession.ModelMetadata.CustomMetadataMap;
            //_customMetadata.Version = encoder_meta.ContainsKey("version") ? encoder_meta["version"] : "";
        }

        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public ModelCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }

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
            var assembly = typeof(VadModel).Assembly;
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

        private ModelCustomMetadata? LoadConf(string configFilePath)
        {
            ModelCustomMetadata? confEntity = new ModelCustomMetadata();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    confEntity = Utils.PreloadHelper.ReadJson(configFilePath);
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confEntity = Utils.PreloadHelper.ReadYaml<ModelCustomMetadata>(configFilePath);
                }
            }
            return confEntity;
        }
    }
}
