using ManySpeech.AudioSep.Model;
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

namespace ManySpeech.AudioSep
{
    internal class SepModel
    {
        private InferenceSession _modelSession;
        private InferenceSession _generatorSession;
        private CustomMetadata _customMetadata;
        private ConfEntity? _confEntity;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;

        public SepModel(string modelFilePath,string generatorFilePath="", string configFilePath = "", int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
            _generatorSession=initModel(generatorFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);

            _customMetadata = new CustomMetadata();
            var encoder_meta = _modelSession.ModelMetadata.CustomMetadataMap;

            string? output_dir = string.Empty;
            encoder_meta.TryGetValue("output_dir", out output_dir);
            _customMetadata.Output_dir = output_dir;

            string? onnx_infer = string.Empty;
            encoder_meta.TryGetValue("onnx.infer", out onnx_infer);
            _customMetadata.Onnx_infer = onnx_infer;

            string? decoder = string.Empty;
            encoder_meta.TryGetValue("decoder", out decoder);
            _customMetadata.Decoder = decoder;

            string? encoder = string.Empty;
            encoder_meta.TryGetValue("encoder", out encoder);
            _customMetadata.Encoder = encoder;

            if (encoder_meta.ContainsKey("output_size"))
            {
                int output_size;
                int.TryParse(encoder_meta["output_size"], out output_size);
                _customMetadata.Output_size = output_size;
            }
            if (encoder_meta.ContainsKey("left_chunks"))
            {
                int left_chunks;
                int.TryParse(encoder_meta["left_chunks"], out left_chunks);
                _customMetadata.Left_chunks = left_chunks;
            }
            if (encoder_meta.ContainsKey("batch"))
            {
                int batch;
                int.TryParse(encoder_meta["batch"], out batch);
                _customMetadata.Batch = batch;
            }
            if (encoder_meta.ContainsKey("reverse_weight"))
            {
                float reverse_weight;
                float.TryParse(encoder_meta["reverse_weight"], out reverse_weight);
                _customMetadata.Reverse_weight = reverse_weight;
            }
            if (encoder_meta.ContainsKey("chunk_size"))
            {
                int chunk_size;
                int.TryParse(encoder_meta["chunk_size"], out chunk_size);
                _customMetadata.Chunk_size = chunk_size;
            }
            if (encoder_meta.ContainsKey("num_blocks"))
            {
                int num_blocks;
                int.TryParse(encoder_meta["num_blocks"], out num_blocks);
                _customMetadata.Num_blocks = num_blocks;
            }
            if (encoder_meta.ContainsKey("cnn_module_kernel"))
            {
                int cnn_module_kernel;
                int.TryParse(encoder_meta["cnn_module_kernel"], out cnn_module_kernel);
                _customMetadata.Cnn_module_kernel = cnn_module_kernel;
            }
            if (encoder_meta.ContainsKey("head"))
            {
                int head;
                int.TryParse(encoder_meta["head"], out head);
                _customMetadata.Head = head;
            }
            if (encoder_meta.ContainsKey("eos_symbol"))
            {
                int eos_symbol;
                int.TryParse(encoder_meta["eos_symbol"], out eos_symbol);
                _customMetadata.Eos_symbol = eos_symbol;
            }
            if (encoder_meta.ContainsKey("feature_size"))
            {
                int feature_size;
                int.TryParse(encoder_meta["feature_size"], out feature_size);
                _customMetadata.Feature_size = feature_size;
            }
            if (encoder_meta.ContainsKey("vocab_size"))
            {
                int vocab_size;
                int.TryParse(encoder_meta["vocab_size"], out vocab_size);
                _customMetadata.Vocab_size = vocab_size;
            }
            if (encoder_meta.ContainsKey("decoding_window"))
            {
                int decoding_window;
                int.TryParse(encoder_meta["decoding_window"], out decoding_window);
                _customMetadata.Decoding_window = decoding_window;
            }
            if (encoder_meta.ContainsKey("subsampling_rate"))
            {
                int subsampling_rate;
                int.TryParse(encoder_meta["subsampling_rate"], out subsampling_rate);
                _customMetadata.Subsampling_rate = subsampling_rate;
            }
            if (encoder_meta.ContainsKey("right_context"))
            {
                int right_context;
                int.TryParse(encoder_meta["right_context"], out right_context);
                _customMetadata.Right_context = right_context;
            }
            if (encoder_meta.ContainsKey("sos_symbol"))
            {
                int sos_symbol;
                int.TryParse(encoder_meta["sos_symbol"], out sos_symbol);
                _customMetadata.Sos_symbol = sos_symbol;
            }
            if (encoder_meta.ContainsKey("is_bidirectional_decoder"))
            {
                bool is_bidirectional_decoder;
                bool.TryParse(encoder_meta["is_bidirectional_decoder"], out is_bidirectional_decoder);
                _customMetadata.Is_bidirectional_decoder = is_bidirectional_decoder;
            }

            if (_customMetadata.Left_chunks <= 0)
            {
                if (_customMetadata.Left_chunks < 0)
                {
                    _required_cache_size = 0;//-1;//
                }
                else
                {
                    _required_cache_size = 0;
                }
            }
            else
            {
                _required_cache_size = _customMetadata.Chunk_size * _customMetadata.Left_chunks;
            }
            _chunkLength = (_customMetadata.Chunk_size - 1) * _customMetadata.Subsampling_rate +
           _customMetadata.Right_context + 1;// Add current frame //_customMetadata.Decoding_window
            _shiftLength = _customMetadata.Subsampling_rate * _customMetadata.Chunk_size;

            _chunkLength = _chunkLength <= 0 ? _customMetadata.Decoding_window : _chunkLength;
            _shiftLength = _shiftLength <= 0 ? _chunkLength : _shiftLength;
        }

        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public InferenceSession GeneratorSession { get => _generatorSession;set=> _generatorSession = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }
        public ConfEntity? ConfEntity { get => _confEntity; set => _confEntity = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Channels { get => _channels; set => _channels = value; }

        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confJsonEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath);
                }
            }
            return confJsonEntity;
        }

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
            if (threadsNum > 0)
                options.InterOpNumThreads = threadsNum;
            else
                options.InterOpNumThreads = System.Environment.ProcessorCount;
            // 启用CPU内存计划
            options.EnableMemoryPattern = true;
            // 设置其他优化选项            
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;            

            InferenceSession onnxSession = null;
            if (!string.IsNullOrEmpty(modelFilePath) && modelFilePath.IndexOf("/") < 0)
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
            var assembly = typeof(SepModel).Assembly;
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
    }
}
