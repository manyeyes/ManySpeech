// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.ASR.Model;
using Microsoft.ML.OnnxRuntime;
//using System.Reflection;

namespace ManySpeech.ASR
{
    public class OfflineModel : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession? _modelSession;
        private InferenceSession? _encoderSession;
        private InferenceSession? _decoderSession;
        private InferenceSession? _embedSession;
        private InferenceSession? _adaptorSession;
        private string? _mvnFilePath;
        private string _tokensFilePath;
        private ConfEntity _confEntity;
        private string[]? _hotwords = null;
        //
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;
        //
        private int _firstLangId = 7; //<ab>
        private int _lastLangId = 144; //<zu>
        private int _firstRegionId = 145; //<AD>
        private int _lastRegionId = 323; //<ZA>
        private int _sosId = 39999; //<sos>
        private int _eosId = 40000; //<sos>
        private int _asrId = 6; //<asr>
        //
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private bool _useITN = false;
        //
        private string _method = "greedy";
        private int _beamWidth = 3;

        public OfflineModel(string tokensFilePath, string configFilePath, string? modelFilePath, string? mvnFilePath = null, string? hotwordFilePath = null, string? embedFilePath = null, int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _embedSession = initModel(embedFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);
            if (_confEntity != null)
            {
                _useITN = _confEntity.use_itn;
            }
            if (!string.IsNullOrEmpty(hotwordFilePath))
            {
                _hotwords = GetHotwords(hotwordFilePath);
            }
            _mvnFilePath = mvnFilePath;
            _tokensFilePath = tokensFilePath;
        }
        public OfflineModel(string tokensFilePath, string configFilePath, string? encoderFilePath, string? decoderFilePath, string? adaptorFilePath = null, string? embedFilePath = null, string? mvnFilePath = null, string? hotwordFilePath = null, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _embedSession = initModel(embedFilePath, threadsNum);
            _adaptorSession = initModel(adaptorFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);
            if (_confEntity != null)
            {
                _useITN = _confEntity.use_itn;
            }
            if (!string.IsNullOrEmpty(hotwordFilePath))
            {
                _hotwords = GetHotwords(hotwordFilePath);
            }
            _mvnFilePath = mvnFilePath;
            _tokensFilePath = tokensFilePath;
        }

        public InferenceSession? ModelSession { get => _modelSession; set => _modelSession = value; }
        public InferenceSession? EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession? AdaptorSession { get => _adaptorSession; set => _adaptorSession = value; }
        public InferenceSession? DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession? EmbedSession { get => _embedSession; set => _embedSession = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public bool UseITN { get => _useITN; set => _useITN = value; }
        public string[]? Hotwords { get => _hotwords; set => _hotwords = value; }
        public ConfEntity ConfEntity { get => _confEntity; set => _confEntity = value; }
        public string? MvnFilePath { get => _mvnFilePath; set => _mvnFilePath = value; }
        public string Method { get => _method; set => _method = value; }
        public int BeamWidth { get => _beamWidth; set => _beamWidth = value; }
        public string TokensFilePath { get => _tokensFilePath; set => _tokensFilePath = value; }
        public int FirstLangId { get => _firstLangId; set => _firstLangId = value; }
        public int LastLangId { get => _lastLangId; set => _lastLangId = value; }
        public int FirstRegionId { get => _firstRegionId; set => _firstRegionId = value; }
        public int LastRegionId { get => _lastRegionId; set => _lastRegionId = value; }
        public int SosId { get => _sosId; set => _sosId = value; }
        public int EosId { get => _eosId; set => _eosId = value; }
        public int AsrId { get => _asrId; set => _asrId = value; }

        internal InferenceSession initModel(string modelFilePath, int threadsNum = 2)
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
            {
                options.IntraOpNumThreads = threadsNum;
                options.InterOpNumThreads = Math.Max(1, Math.Min(threadsNum / 2, 4));
            }
            else
            {
                options.IntraOpNumThreads = System.Environment.ProcessorCount;
            }
            // 启用CPU内存计划
            options.EnableMemoryPattern = true;
            // 设置其他优化选项            
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            InferenceSession onnxSession = null;
            if (modelFilePath.IndexOf("/") < 0 && modelFilePath.IndexOf("\\") < 0)
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
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath); // To compile for AOT
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
                }
            }
            return confJsonEntity;
        }
        private string[]? GetHotwords(string hotwordFilePath)
        {
            string[]? hotwords = null;
            if (File.Exists(hotwordFilePath))
            {
                hotwords = File.ReadAllLines(hotwordFilePath);
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
                    if (_encoderSession != null)
                    {
                        _encoderSession.Dispose();
                    }
                    if (_decoderSession != null)
                    {
                        _decoderSession.Dispose();
                    }
                    if (_adaptorSession != null)
                    {
                        _adaptorSession.Dispose();
                    }
                    if (_embedSession != null)
                    {
                        _embedSession.Dispose();
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
