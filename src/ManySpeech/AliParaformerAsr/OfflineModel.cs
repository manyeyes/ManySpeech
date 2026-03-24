// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;
using YamlDotNet.Core.Tokens;
//using System.Reflection;

namespace ManySpeech.AliParaformerAsr
{
    public class OfflineModel : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession? _modelSession;
        private InferenceSession? _embedSession;
        private string? _mvnFilePath;
        private string[]? _tokens;
        private ConfEntity? _confEntity;
        private List<int[]>? _hotwords = null;

        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private bool _use_itn = false;

        private string _method = "greedy";
        private int _beamWidth = 3;

        public OfflineModel( string tokensFilePath, string configFilePath, string? modelFilePath,string? mvnFilePath = null, string? hotwordFilePath = null, string? embedFilePath = null, int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _embedSession = initModel(embedFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);
            if (_confEntity != null)
            {
                _use_itn = _confEntity.use_itn;
            }
            _mvnFilePath = mvnFilePath;
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
            if (_tokens == null || _tokens.Length == 0)
            {
                throw new Exception("tokens invalid");
            }
            if (!string.IsNullOrEmpty(hotwordFilePath))
            {
                List<int[]>? hotwords = GetHotwords(_tokens, hotwordFilePath);
                _hotwords = hotwords;
            }
        }

        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public InferenceSession? EmbedSession { get => _embedSession; set => _embedSession = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public bool Use_itn { get => _use_itn; set => _use_itn = value; }
        public List<int[]>? Hotwords { get => _hotwords; set => _hotwords = value; }
        public ConfEntity ConfEntity { get => _confEntity; set => _confEntity = value; }
        public string[] Tokens { get => _tokens; set => _tokens = value; }
        public string MvnFilePath { get => _mvnFilePath; set => _mvnFilePath = value; }
        public string Method { get => _method; set => _method = value; }
        public int BeamWidth { get => _beamWidth; set => _beamWidth = value; }

        public InferenceSession? initModel(string? modelFilePath, int threadsNum = 2)
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

        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confJsonEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confJsonEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath); // To compile for AOT
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
                }
            }
            return confJsonEntity;
        }
        private List<int[]>? GetHotwords(string[] tokens, string hotwordFilePath)
        {
            List<int[]>? hotwords = new List<int[]>();
            if (File.Exists(hotwordFilePath))
            {
                string[] sentences = File.ReadAllLines(hotwordFilePath);
                foreach (string sentence in sentences)
                {
                    string[] wordList = new string[] { sentence };//TODO:分词
                    foreach (string word in wordList)
                    {
                        List<int> ids = word.ToCharArray().Select(x => Array.IndexOf(tokens, x.ToString())).Where(x => x != -1).ToList();
                        hotwords.Add(ids.ToArray());
                    }
                }
                hotwords.Add(new int[] { _sos_eos_id });
            }
            return hotwords;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_modelSession != null)
                    {
                        _modelSession.Dispose();
                    }
                    if (_tokens != null)
                    {
                        _tokens = null;
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
