using ManySpeech.AudioSep.Model;
using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using System.Reflection;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// Separates audio using ONNX models, handling model loading, configuration, and inference setup
    /// </summary>
    internal class SepModel : IDisposable
    {
        private InferenceSession _modelSession;
        private InferenceSession _generatorSession;
        private CustomMetadata _customMetadata;
        private ConfEntity _confEntity;

        private int _featureDimension = 80;
        private int _sampleRate = 16000;
        private int _channels = 1;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _requiredCacheSize = 0;

        /// <summary>
        /// Gets the main model inference session
        /// </summary>
        public InferenceSession ModelSession
        {
            get => _modelSession;
            set => _modelSession = value;
        }

        /// <summary>
        /// Gets the generator model inference session
        /// </summary>
        public InferenceSession GeneratorSession
        {
            get => _generatorSession;
            set => _generatorSession = value;
        }

        /// <summary>
        /// Gets the length of audio chunks processed in each step
        /// </summary>
        public int ChunkLength
        {
            get => _chunkLength;
            set => _chunkLength = value;
        }

        /// <summary>
        /// Gets the shift length between consecutive chunks
        /// </summary>
        public int ShiftLength
        {
            get => _shiftLength;
            set => _shiftLength = value;
        }

        /// <summary>
        /// Gets the dimension of input features
        /// </summary>
        public int FeatureDimension
        {
            get => _featureDimension;
            set => _featureDimension = value;
        }

        /// <summary>
        /// Gets the audio sample rate
        /// </summary>
        public int SampleRate
        {
            get => _sampleRate;
            set => _sampleRate = value;
        }

        /// <summary>
        /// Gets the required cache size for processing
        /// </summary>
        public int RequiredCacheSize
        {
            get => _requiredCacheSize;
            set => _requiredCacheSize = value;
        }

        /// <summary>
        /// Gets the configuration entity
        /// </summary>
        public ConfEntity ConfEntity
        {
            get => _confEntity;
            set => _confEntity = value;
        }

        /// <summary>
        /// Gets custom metadata from the model
        /// </summary>
        public CustomMetadata CustomMetadata
        {
            get => _customMetadata;
            set => _customMetadata = value;
        }

        /// <summary>
        /// Gets the number of audio channels
        /// </summary>
        public int Channels
        {
            get => _channels;
            set => _channels = value;
        }

        /// <summary>
        /// Initializes a new instance of the SepModel class
        /// </summary>
        /// <param name="modelFilePath">Path to the main model file</param>
        /// <param name="generatorFilePath">Path to the generator model file (optional)</param>
        /// <param name="configFilePath">Path to the configuration file (optional)</param>
        /// <param name="threadsNum">Number of threads to use for inference</param>
        public SepModel(string modelFilePath, string generatorFilePath = "", string configFilePath = "", int threadsNum = 2)
        {
            _modelSession = InitializeModel(modelFilePath, threadsNum);
            _generatorSession = InitializeModel(generatorFilePath, threadsNum);
            _confEntity = LoadConfiguration(configFilePath);
            _customMetadata = new CustomMetadata();

            //LoadCustomMetadata(_modelSession.ModelMetadata.CustomMetadataMap);
            //CalculateProcessingParameters();
        }

        ///// <summary>
        ///// Loads custom metadata from the model's metadata map
        ///// </summary>
        ///// <param name="metadataMap">Model metadata map containing custom key-value pairs</param>
        //private void LoadCustomMetadata(IReadOnlyDictionary<string, string> metadataMap)
        //{
        //    // Load string metadata
        //    _customMetadata.OutputDir = GetMetadataValue(metadataMap, "output_dir");
        //    _customMetadata.OnnxInfer = GetMetadataValue(metadataMap, "onnx.infer");
        //    _customMetadata.Decoder = GetMetadataValue(metadataMap, "decoder");
        //    _customMetadata.Encoder = GetMetadataValue(metadataMap, "encoder");

        //    // Load integer metadata
        //    _customMetadata.OutputSize = GetMetadataIntValue(metadataMap, "output_size");
        //    _customMetadata.LeftChunks = GetMetadataIntValue(metadataMap, "left_chunks");
        //    _customMetadata.Batch = GetMetadataIntValue(metadataMap, "batch");
        //    _customMetadata.ChunkSize = GetMetadataIntValue(metadataMap, "chunk_size");
        //    _customMetadata.NumBlocks = GetMetadataIntValue(metadataMap, "num_blocks");
        //    _customMetadata.CnnModuleKernel = GetMetadataIntValue(metadataMap, "cnn_module_kernel");
        //    _customMetadata.Head = GetMetadataIntValue(metadataMap, "head");
        //    _customMetadata.EosSymbol = GetMetadataIntValue(metadataMap, "eos_symbol");
        //    _customMetadata.FeatureSize = GetMetadataIntValue(metadataMap, "feature_size");
        //    _customMetadata.VocabSize = GetMetadataIntValue(metadataMap, "vocab_size");
        //    _customMetadata.DecodingWindow = GetMetadataIntValue(metadataMap, "decoding_window");
        //    _customMetadata.SubsamplingRate = GetMetadataIntValue(metadataMap, "subsampling_rate");
        //    _customMetadata.RightContext = GetMetadataIntValue(metadataMap, "right_context");
        //    _customMetadata.SosSymbol = GetMetadataIntValue(metadataMap, "sos_symbol");

        //    // Load floating-point metadata
        //    _customMetadata.ReverseWeight = GetMetadataFloatValue(metadataMap, "reverse_weight");

        //    // Load boolean metadata
        //    _customMetadata.IsBidirectionalDecoder = GetMetadataBoolValue(metadataMap, "is_bidirectional_decoder");
        //}

        /// <summary>
        /// Calculates processing parameters based on custom metadata
        /// </summary>
        //private void CalculateProcessingParameters()
        //{
        //    // Determine required cache size
        //    if (_customMetadata.LeftChunks <= 0)
        //    {
        //        _requiredCacheSize = _customMetadata.LeftChunks < 0 ? 0 : 0;
        //    }
        //    else
        //    {
        //        _requiredCacheSize = _customMetadata.ChunkSize * _customMetadata.LeftChunks;
        //    }

        //    // Calculate chunk and shift lengths
        //    _chunkLength = (_customMetadata.ChunkSize - 1) * _customMetadata.SubsamplingRate +
        //                  _customMetadata.RightContext + 1; // Include current frame

        //    _shiftLength = _customMetadata.SubsamplingRate * _customMetadata.ChunkSize;

        //    // Fallback to decoding window if chunk length is invalid
        //    _chunkLength = _chunkLength <= 0 ? _customMetadata.DecodingWindow : _chunkLength;
        //    _shiftLength = _shiftLength <= 0 ? _chunkLength : _shiftLength;
        //}

        /// <summary>
        /// Loads configuration from a JSON file
        /// </summary>
        /// <param name="configFilePath">Path to the configuration file</param>
        /// <returns>Loaded configuration entity or null if file not found</returns>
        private ConfEntity LoadConfiguration(string configFilePath)
        {
            if (string.IsNullOrEmpty(configFilePath) || !File.Exists(configFilePath))
                return new ConfEntity();

            return configFilePath.ToLower().EndsWith(".json")
                ? Utils.PreloadHelper.ReadJson(configFilePath)
                : new ConfEntity();
        }

        /// <summary>
        /// Initializes an ONNX inference session
        /// </summary>
        /// <param name="modelFilePath">Path to the ONNX model file</param>
        /// <param name="threadsNum">Number of threads to use</param>
        /// <returns>Initialized inference session or null if model not found</returns>
        public InferenceSession InitializeModel(string modelFilePath, int threadsNum = 2)
        {
            if (string.IsNullOrEmpty(modelFilePath) || !File.Exists(modelFilePath))
                return null;

            var options = new SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableMemoryPattern = true // Enable CPU memory planning
            };

            // Configure execution provider (CPU)
            options.AppendExecutionProvider_CPU(0);

            // Configure thread count
            options.InterOpNumThreads = threadsNum > 0 ? threadsNum : Environment.ProcessorCount;

            // Load model from embedded resource or file system
            return modelFilePath.IndexOf("/") < 0
                ? new InferenceSession(ReadEmbeddedResourceAsBytes(modelFilePath), options)
                : new InferenceSession(modelFilePath, options);
        }

        /// <summary>
        /// Reads an embedded resource as a byte array
        /// </summary>
        /// <param name="resourceName">Name of the embedded resource</param>
        /// <returns>Byte array containing the resource data</returns>
        /// <exception cref="FileNotFoundException">Thrown if the resource is not found</exception>
        private static byte[] ReadEmbeddedResourceAsBytes(string resourceName)
        {
            var assembly = typeof(SepModel).Assembly;
            using var stream = assembly.GetManifestResourceStream(resourceName)
                ?? throw new FileNotFoundException($"Embedded resource '{resourceName}' not found.");

            byte[] bytes = new byte[stream.Length];
            stream.Read(bytes, 0, bytes.Length);
            return bytes;
        }

        /// <summary>
        /// Gets a string value from the metadata map
        /// </summary>
        /// <param name="metadataMap">Metadata map</param>
        /// <param name="key">Metadata key</param>
        /// <returns>Metadata value or empty string if key not found</returns>
        private string GetMetadataValue(IReadOnlyDictionary<string, string> metadataMap, string key)
        {
            metadataMap.TryGetValue(key, out string value);
            return value ?? string.Empty;
        }

        /// <summary>
        /// Gets an integer value from the metadata map
        /// </summary>
        /// <param name="metadataMap">Metadata map</param>
        /// <param name="key">Metadata key</param>
        /// <returns>Parsed integer or 0 if key not found or parsing fails</returns>
        private int GetMetadataIntValue(IReadOnlyDictionary<string, string> metadataMap, string key)
        {
            if (metadataMap.TryGetValue(key, out string value) && int.TryParse(value, out int result))
                return result;
            return 0;
        }

        /// <summary>
        /// Gets a float value from the metadata map
        /// </summary>
        /// <param name="metadataMap">Metadata map</param>
        /// <param name="key">Metadata key</param>
        /// <returns>Parsed float or 0 if key not found or parsing fails</returns>
        private float GetMetadataFloatValue(IReadOnlyDictionary<string, string> metadataMap, string key)
        {
            if (metadataMap.TryGetValue(key, out string value) && float.TryParse(value, out float result))
                return result;
            return 0f;
        }

        /// <summary>
        /// Gets a boolean value from the metadata map
        /// </summary>
        /// <param name="metadataMap">Metadata map</param>
        /// <param name="key">Metadata key</param>
        /// <returns>Parsed boolean or false if key not found or parsing fails</returns>
        private bool GetMetadataBoolValue(IReadOnlyDictionary<string, string> metadataMap, string key)
        {
            if (metadataMap.TryGetValue(key, out string value) && bool.TryParse(value, out bool result))
                return result;
            return false;
        }

        /// <summary>
        /// Disposes of the inference sessions
        /// </summary>
        public void Dispose()
        {
            _modelSession?.Dispose();
            _generatorSession?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}