// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime;
using System.Reflection;

namespace ManySpeech.K2TransducerAsr
{
    /// <summary>
    /// online model
    /// </summary>
    public class OnlineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OnlineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        public OnlineModel(string encoderFilePath, string decoderFilePath, string joinerFilePath, string configFilePath = "", int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _joinerSession = initModel(joinerFilePath, threadsNum);

            if (_encoderSession != null)
            {
                var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

                _customMetadata = new OnlineCustomMetadata();
                int decode_chunk_len = 0;
                if (encoder_meta.ContainsKey("decode_chunk_len"))
                {
                    int.TryParse(encoder_meta["decode_chunk_len"].ToString(), out decode_chunk_len);
                    _customMetadata.Decode_chunk_len = decode_chunk_len;
                }

                int _TT;
                int.TryParse(encoder_meta["T"].ToString(), out _TT);
                _customMetadata.TT = _TT;

                _chunkLength = _TT;
                _shiftLength = decode_chunk_len;

                if (encoder_meta.ContainsKey("num_encoder_layers"))
                {
                    _customMetadata.Num_encoder_layers = Array.ConvertAll(encoder_meta["num_encoder_layers"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("encoder_dims"))
                {
                    _customMetadata.Encoder_dims = Array.ConvertAll(encoder_meta["encoder_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("attention_dims"))
                {
                    _customMetadata.Attention_dims = Array.ConvertAll(encoder_meta["attention_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("cnn_module_kernels"))
                {
                    _customMetadata.Cnn_module_kernels = Array.ConvertAll(encoder_meta["cnn_module_kernels"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("left_context_len"))
                {
                    _customMetadata.Left_context_len = Array.ConvertAll(encoder_meta["left_context_len"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }

                string? model_type = string.Empty;
                encoder_meta.TryGetValue("model_type", out model_type);
                _customMetadata.Model_type = model_type;
                string? model_author = string.Empty;
                encoder_meta.TryGetValue("model_author", out model_author);
                _customMetadata.Model_author = model_author;
                string? version = string.Empty;
                encoder_meta.TryGetValue("version", out version);
                _customMetadata.Version = version;

                //zipformer2
                if (encoder_meta.ContainsKey("query_head_dims"))
                {
                    _customMetadata.Query_head_dims = Array.ConvertAll(encoder_meta["query_head_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("value_head_dims"))
                {
                    _customMetadata.Value_head_dims = Array.ConvertAll(encoder_meta["value_head_dims"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                if (encoder_meta.ContainsKey("num_heads"))
                {
                    _customMetadata.Num_heads = Array.ConvertAll(encoder_meta["num_heads"].ToString().Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries), s => int.TryParse(s, out int i) ? i : 0);
                }
                string? onnx_infer = string.Empty;
                encoder_meta.TryGetValue("onnx.infer", out onnx_infer);
                _customMetadata.Onnx_infer = onnx_infer;
                string? comment = string.Empty;
                encoder_meta.TryGetValue("comment", out comment);
                _customMetadata.Comment = comment;
                if (!string.IsNullOrEmpty(comment))
                {
                    if (comment.Contains("ctc") && comment.Contains("zipformer2"))
                    {
                        _customMetadata.Model_type = model_type + "ctc";
                    }
                }
                if (encoder_meta.ContainsKey("feature"))
                {
                    string? feature_type = "fbank";
                    encoder_meta.TryGetValue("feature", out feature_type);
                    if (!string.IsNullOrEmpty(feature_type))
                    {
                        _customMetadata.Feature_type = feature_type;
                    }
                }
                //lstm
                if (encoder_meta.ContainsKey("d_model"))
                {
                    int d_model;
                    int.TryParse(encoder_meta["d_model"], out d_model);
                    _customMetadata.D_model = d_model;
                }
                if (encoder_meta.ContainsKey("rnn_hidden_size"))
                {
                    int rnn_hidden_size;
                    int.TryParse(encoder_meta["rnn_hidden_size"], out rnn_hidden_size);
                    _customMetadata.Rnn_hidden_size = rnn_hidden_size;
                }
                //conformer
                if (encoder_meta.ContainsKey("cnn_module_kernel"))
                {
                    int cnn_module_kernel;
                    int.TryParse(encoder_meta["cnn_module_kernel"], out cnn_module_kernel);
                    _customMetadata.Cnn_module_kernel = cnn_module_kernel;
                }
                if (encoder_meta.ContainsKey("pad_length"))
                {
                    int pad_length;
                    int.TryParse(encoder_meta["pad_length"], out pad_length);
                    _customMetadata.Pad_length = pad_length;
                }
                if (encoder_meta.ContainsKey("encoder_dim"))
                {
                    int encoder_dim;
                    int.TryParse(encoder_meta["encoder_dim"], out encoder_dim);
                    _customMetadata.Encoder_dim = encoder_dim;
                }
                if (encoder_meta.ContainsKey("chunk_size"))
                {
                    int chunk_size;
                    int.TryParse(encoder_meta["chunk_size"], out chunk_size);
                    _customMetadata.Chunk_size = chunk_size;
                }
                if (encoder_meta.ContainsKey("left_context"))
                {
                    int left_context;
                    int.TryParse(encoder_meta["left_context"], out left_context);
                    _customMetadata.Left_context = left_context;
                }
                if (encoder_meta.ContainsKey("right_context"))
                {
                    int right_context;
                    int.TryParse(encoder_meta["right_context"], out right_context);
                    _customMetadata.Right_context = right_context;
                }
            }

            if (_decoderSession != null)
            {
                int context_size;
                int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["context_size"], out context_size);
                CustomMetadata.Context_size = context_size;
                int vocab_size;
                int.TryParse(_decoderSession.ModelMetadata.CustomMetadataMap["vocab_size"], out vocab_size);
                CustomMetadata.Vocab_size = vocab_size;
            }
            if (_joinerSession != null)
            {
                int joiner_dim;
                int.TryParse(_joinerSession.ModelMetadata.CustomMetadataMap["joiner_dim"], out joiner_dim);
                _customMetadata.Joiner_dim = joiner_dim;
            }
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OnlineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

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

    }
}
