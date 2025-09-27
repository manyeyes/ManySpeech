// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using ManySpeech.WenetAsr2.Model;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.WenetAsr2
{
    public class OfflineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _ctcSession;
        private CustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _sos_eos_id = 1;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;

        public OfflineModel(string encoderFilePath, string decoderFilePath, string ctcFilePath, string configFilePath="", int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _ctcSession = initModel(ctcFilePath, threadsNum);

            _customMetadata = new CustomMetadata();
            var encoder_meta = _encoderSession.ModelMetadata.CustomMetadataMap;

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
            _sos_eos_id = _customMetadata.Sos_symbol;

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
           _customMetadata.Right_context + 1;//# Add current frame //_customMetadata.Decoding_window
            _shiftLength = _customMetadata.Subsampling_rate * _customMetadata.Chunk_size;

            _chunkLength = _chunkLength <= 0 ? _customMetadata.Decoding_window : _chunkLength;
            _shiftLength = _shiftLength <= 0 ? _chunkLength : _shiftLength;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession CtcSession { get => _ctcSession; set => _ctcSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }

        public InferenceSession initModel(string modelFilePath, int threadsNum = 2)
        {
            Microsoft.ML.OnnxRuntime.SessionOptions options = new Microsoft.ML.OnnxRuntime.SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            //options.AppendExecutionProvider_DML(0);
            options.AppendExecutionProvider_CPU(0);
            //options.AppendExecutionProvider_CUDA(0);
            options.InterOpNumThreads = threadsNum;
            InferenceSession onnxSession = new InferenceSession(modelFilePath, options);
            return onnxSession;
        }

        
        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            states = statesList[0];
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 2 == 0, "when stack_states, state_list[0] is 2x");
            statesList.Add(states);
            return statesList;
        }
    }
}
