// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
//using System.Reflection;

namespace ManySpeech.AliParaformerAsr
{
    public class EmbedSeacoModel
    {
        private InferenceSession _modelSession;

        public EmbedSeacoModel(string modelFilePath, int threadsNum = 2)
        {
            _modelSession = initModel(modelFilePath, threadsNum);
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }

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
            var assembly = typeof(EmbedSeacoModel).Assembly;
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
        public Tensor<float>? Forward(List<int[]>? hotwords)
        {
            if (hotwords == null || hotwords.Count == 0)
            {
                return null;
            }
            //float[] y=new float[0];
            Tensor<float>? hwEmbed = null;
            int numHotwords = hotwords.Count;
            int maxLength = 10;
            int[] hotwords_pad = PadList(hotwords, 0, maxLength);
            var inputMeta = _modelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hotword")
                {
                    int[] dim = new int[] { numHotwords, 10 };
                    var tensor = new DenseTensor<int>(hotwords_pad, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
            }
            //IReadOnlyCollection<string> outputNames = new List<string>();
            //outputNames.Append("hw_embed");
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
            try
            {
                results = _modelSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    hwEmbed = resultsArray[0].AsTensor<float>();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Embed SeACo Forward failed", ex.InnerException);
            }
            return hwEmbed;
        }
        private int[] PadList(List<int[]> hotwords, int paddingValue, int maxLength = 0)
        {
            List<int[]> hotwordsPadList = new List<int[]>(hotwords);
            if (maxLength == 0)
            {
                maxLength = hotwords.Select(x => x.Length).Max();
            }
            for (int i = 0; i < hotwordsPadList.Count; i++)
            {
                hotwordsPadList[i] = hotwordsPadList[i].Length > maxLength ? hotwordsPadList[i].Take(maxLength).ToArray() : hotwordsPadList[i].Concat(Enumerable.Repeat(paddingValue, maxLength - hotwordsPadList[i].Length)).ToArray();
            }
            int[] hotwordsPad = hotwordsPadList.SelectMany(x => x).ToArray();
            return hotwordsPad;
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
