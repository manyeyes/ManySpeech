// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
//using System.Reflection;

namespace ManySpeech.AliParaformerAsr
{
    public class OnlineModel
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;

        private int _chunkSize = 5;
        private int _lfr = 10;
        private int _chunkLength;
        private int _shiftLength;
        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private string _mvnFilePath;
        private ConfEntity? _confEntity;

        public OnlineModel(string encoderFilePath, string decoderFilePath, string mvnFilePath, string configFilePath, int threadsNum = 2)
        {
            _encoderSession = initModel(encoderFilePath, threadsNum);
            _decoderSession = initModel(decoderFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);
            _mvnFilePath = mvnFilePath;
            _chunkLength = _lfr * _chunkSize + 10;
            _shiftLength = _chunkLength;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public string MvnFilePath { get => _mvnFilePath; set => _mvnFilePath = value; }
        public ConfEntity? ConfEntity { get => _confEntity; set => _confEntity = value; }

        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confJsonEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confJsonEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath);
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
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
            var assembly = typeof(OnlineModel).Assembly;
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
        public List<float[]> StackCifHiddens(List<List<float[]>> cifHiddens)
        {
            int batchSize = cifHiddens.Count;
            List<float[]> cifHidden = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                foreach (float[] item in cifHiddens[b])
                {
                    cifHidden.Add(item);
                }
            }
            return cifHidden;
        }

        public List<List<float[]>> UnStackCifHiddens(List<float[]> cifHidden, int batchSize)
        {
            List<List<float[]>> cifHiddens = new List<List<float[]>>();
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> hiddensItem = new List<float[]>();
                for (int x = b * (cifHidden.Count / batchSize); x < (b + 1) * (cifHidden.Count / batchSize); x++)
                {
                    hiddensItem.Add(cifHidden[x]);
                }
                cifHiddens.Add(hiddensItem);
            }
            return cifHiddens;
        }

        public List<List<float>> DynamicMask(List<List<float>> alphas)
        {
            List<List<float>> newAlphas = new List<List<float>>();
            foreach (List<float> item in alphas)
            {
                float[] cifAlphasItem = item.ToArray();
                float[] chunk_size_5 = new float[_chunkSize];
                if (cifAlphasItem.Length > chunk_size_5.Length)
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, chunk_size_5.Length);
                }
                else
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, cifAlphasItem.Length);
                }
                int decodeLfr = chunk_size_5.Length + _lfr;
                if (cifAlphasItem.Length > decodeLfr)
                {
                    float[] chunk_size_15 = new float[cifAlphasItem.Length - decodeLfr];
                    Array.Copy(chunk_size_15, 0, cifAlphasItem, decodeLfr, chunk_size_15.Length);
                }
                newAlphas.Add(cifAlphasItem.ToList());
            }
            return newAlphas;
        }

        public float[] StackCifAlphas(List<float[]> cifAlphaList)
        {
            int batchSize = cifAlphaList.Count;
            float[] cifAlphas = new float[cifAlphaList[0].Length * batchSize];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(cifAlphaList[b], 0, cifAlphas, b * cifAlphaList[0].Length, cifAlphaList[b].Length);
            }
            return cifAlphas;
        }

        public List<float[]> UnStackCifAlphas(float[] cifAlphas, int batchSize)
        {
            List<float[]> cifAlphaList = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                float[] cifAlphasItem = new float[cifAlphas.Length / batchSize];
                Array.Copy(cifAlphas, b * cifAlphasItem.Length, cifAlphasItem, 0, cifAlphasItem.Length);
                //////////
                float[] chunk_size_5 = new float[_chunkSize];
                if (cifAlphasItem.Length > _chunkSize)
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, chunk_size_5.Length);
                }
                else
                {
                    Array.Copy(chunk_size_5, 0, cifAlphasItem, 0, cifAlphasItem.Length);
                }
                int decodeLfr = chunk_size_5.Length + _lfr;
                if (cifAlphasItem.Length > decodeLfr)
                {
                    float[] chunk_size_15 = new float[cifAlphasItem.Length - decodeLfr];
                    Array.Copy(chunk_size_15, 0, cifAlphasItem, decodeLfr, chunk_size_15.Length);
                }
                //////////
                cifAlphaList.Add(cifAlphasItem);
            }
            return cifAlphaList;
        }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            int batchSize = statesList.Count;
            Debug.Assert(statesList[0].Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = statesList[0].Count;
            for (int i = 0; i < fsmnLayer; i++)
            {
                float[] statesItemTemp = new float[statesList[0][i].Length * batchSize];
                int statesItemTemp_item_length = statesList[0][i].Length;
                int statesItemTemp_item_axisnum = 512 * 10;
                for (int x = 0; x < statesItemTemp_item_length / statesItemTemp_item_axisnum; x++)
                {
                    for (int n = 0; n < batchSize; n++)
                    {
                        float[] statesItemTemp_item = statesList[n][0];
                        Array.Copy(statesItemTemp_item, x * statesItemTemp_item_axisnum, statesItemTemp, (x * batchSize + n) * statesItemTemp_item_axisnum, statesItemTemp_item_axisnum);
                    }
                }
                states.Add(statesItemTemp);
            }
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 16 == 0, "when stack_states, state_list[0] is 16x");
            int fsmnLayer = states.Count;
            int batchSize = states[0].Length / 512 / 10;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 512 * 10;
                    int statesItemTemp_size = 1 * 512 * 10;
                    float[] statesItemTemp_item = new float[statesItemTemp_size];
                    for (int k = 0; k < statesItemTemp_size / statesItemTemp_axisnum; k++)
                    {
                        Array.Copy(item, (item.Length / statesItemTemp_size * k + b) * statesItemTemp_axisnum, statesItemTemp_item, k * statesItemTemp_axisnum, statesItemTemp_axisnum);
                    }
                    statesListItem.Add(statesItemTemp_item);
                }
                statesList.Add(statesListItem);
            }
            return statesList;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_encoderSession != null)
                {
                    _encoderSession.Dispose();
                }
                if (_decoderSession != null)
                {
                    _decoderSession.Dispose();
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
