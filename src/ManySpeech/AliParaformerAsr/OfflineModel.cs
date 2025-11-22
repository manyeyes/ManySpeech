// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using Microsoft.ML.OnnxRuntime;
//using System.Reflection;

namespace ManySpeech.AliParaformerAsr
{
    public class OfflineModel
    {
        private InferenceSession _modelSession;
        private string _modelebFilePath;
        private List<int[]>? _hotwords = null;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private bool _use_itn = false;

        public OfflineModel(string modelFilePath, string modelebFilePath = "", int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _modelebFilePath = modelebFilePath;
        }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public bool Use_itn { get => _use_itn; set => _use_itn = value; }
        public string ModelebFilePath { get => _modelebFilePath; set => _modelebFilePath = value; }
        public List<int[]>? Hotwords { get => _hotwords; set => _hotwords = value; }

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
            var assembly = typeof(OfflineModel).Assembly;
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
            if (disposing)
            {
                if (_modelSession != null)
                {
                    _modelSession.Dispose();
                }
            }
        }

        internal void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
