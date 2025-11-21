// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ManySpeech.SileroVad.Model;
using ManySpeech.SileroVad.Utils;

namespace ManySpeech.SileroVad
{
    public class VadProj: IVadProj
    {
        private InferenceSession _modelSession;
        private ModelCustomMetadata _customMetadata;
        private bool _isDebug;
        private int _sampleRate = 16000;


        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public ModelCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public bool IsDebug { get => _isDebug; set => _isDebug = value; }

        public VadProj(VadModel vadModel, int sampleRate = 16000,bool isDebug=false)
        {
            _modelSession = vadModel.ModelSession;

            _customMetadata = new ModelCustomMetadata();
            _customMetadata = vadModel.CustomMetadata;
            _sampleRate = sampleRate;
            _isDebug = isDebug;
        }

        public List<float[]> GetModelInitStates(int batchSize = 1)
        {
            int num_model_layers = 2;//_customMetadata.Num_model_layers[i];
            //h
            int h_size = num_model_layers * batchSize * 64;
            float[] h = new float[h_size];
            //c
            int c_size = num_model_layers * batchSize * 64;
            float[] c = new float[h_size];

            float[] sr = new float[1];
            sr[0] = SampleRate;
            List<float[]> states = new List<float[]> { sr, h, c };
            return states;
        }

        public List<float[]> stack_states_unittest(List<List<float[]>> stateList)
        {
            List<float[]> states = new List<float[]>();
            states = stateList[0];
            return states;
        }

        public List<List<float[]>> unstack_states_unittest(List<float[]> stateList)
        {
            float[] sr = new float[1];
            sr[0] = SampleRate;
            stateList.Insert(0, sr);
            List<List<float[]>> xxx = new List<List<float[]>>();
            xxx.Add(stateList);
            return xxx;
        }

        public List<float[]> stack_states(List<List<float[]>> stateList)
        {
            int batchSize = stateList.Count;
            int num_model_layers = 2;//_customMetadata.Num_model_layers[i];
            //h
            List<float[]> h_list = new List<float[]>();
            for (int n = 0; n < batchSize; n++)
            {
                h_list.Add(stateList[n][1]);
            }
            float[] h = new float[h_list[0].Length * batchSize];
            int h_item_length = h_list[0].Length;
            int h_axisnum = 64;
            for (int x = 0; x < h_item_length / h_axisnum; x++)
            {
                for (int n = 0; n < batchSize; n++)
                {
                    float[] h_item = h_list[n];
                    Array.Copy(h_item, x * h_axisnum, h, (x * batchSize + n) * h_axisnum, h_axisnum);
                    //Array.Copy(avg_item, n * avg_item_length + x, avg, x * batch_size + n, 1);
                }
            }
            //c
            List<float[]> c_list = new List<float[]>();
            for (int n = 0; n < batchSize; n++)
            {
                c_list.Add(stateList[n][2]);
            }
            float[] c = new float[c_list[0].Length * batchSize];
            int c_item_length = c_list[0].Length;
            int c_axisnum = 64;
            for (int x = 0; x < c_item_length / c_axisnum; x++)
            {
                for (int n = 0; n < batchSize; n++)
                {
                    float[] c_item = c_list[n];
                    Array.Copy(c_item, x * c_axisnum, c, (x * batchSize + n) * c_axisnum, c_axisnum);
                    //Array.Copy(avg_item, n * avg_item_length + x, avg, x * batcc_size + n, 1);
                }
            }

            float[] sr = new float[1];
            sr[0] = SampleRate;

            List<float[]> states = new List<float[]> { sr, h, c };
            return states;
        }

        public List<List<float[]>> unstack_states(List<float[]> modelOutStates)
        {
            int batchSize = modelOutStates[0].Length / 2 / 64;
            float[] sr = new float[1];
            sr[0] = SampleRate;
            List<List<float[]>> statesList = new List<List<float[]>>();
            int num_model_layers = 2;
            int n = 1;
            for (int i = 0; i < batchSize; i++)
            {
                //sr
                List<float[]> states = new List<float[]>();
                states.Add(sr);
                //hn
                float[] hn = modelOutStates[0];
                int h_axisnum = 64;
                int h_size = num_model_layers * n * h_axisnum;
                float[] h = new float[h_size];
                for (int k = 0; k < h_size / h_axisnum; k++)
                {
                    Array.Copy(hn, (hn.Length / h_size * k + i) * h_axisnum, h, k * h_axisnum, h_axisnum);
                }
                states.Add(h);
                //cn
                float[] cn = modelOutStates[1];
                int c_axisnum = 64;
                int c_size = num_model_layers * n * c_axisnum;
                float[] c = new float[c_size];
                for (int k = 0; k < c_size / c_axisnum; k++)
                {
                    Array.Copy(cn, (cn.Length / c_size * k + i) * c_axisnum, c, k * c_axisnum, c_axisnum);
                }
                states.Add(c);
                statesList.Add(states);
            }
            return statesList;
        }

        public ModelOutputEntity ModelProj(List<ModelInputEntity> modelInputs, List<float[]> statesList)
        {
            int batchSize = modelInputs.Count;
            int sampleRate = (int)statesList[0][0];
            int contextSize = sampleRate == 16000 ? 64 : 32;
            modelInputs = modelInputs.Select(x => { x.Speech = new float[contextSize].Select(x => x = float.PositiveInfinity).Concat(x.Speech.Select((float x) => x * 32768f)).ToArray(); x.SpeechLength += contextSize; return x; }).ToList();
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            padSequence = padSequence.Select(x => x == float.PositiveInfinity ? 0f : x).ToArray();
            var inputMeta = _modelSession.InputMetadata;
            ModelOutputEntity modelOutput = new ModelOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / batchSize };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "sr")
                {
                    int[] dim = new int[] { 1 };
                    Int64[] sr = new Int64[1] { sampleRate };
                    var tensor = new DenseTensor<Int64>(sr, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "h")
                {
                    int[] dim = new int[] { 2, batchSize, 64 };
                    var tensor = new DenseTensor<float>(statesList[1], dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "c")
                {
                    int[] dim = new int[] { 2, batchSize, 64 };
                    var tensor = new DenseTensor<float>(statesList[2], dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;
                results = _modelSession.Run(container);
                var resultsArray = results.ToArray();
                modelOutput.ModelOutStates = new List<float[]>();
                float[]? hn = null;
                float[]? cn = null;
                for (int j = 0; j < resultsArray.Length; j++)
                {
                    if (resultsArray[j].Name.Equals("output"))
                    {
                        modelOutput.ModelOut = resultsArray[j].AsEnumerable<float>().ToArray();
                    }
                    if (resultsArray[j].Name.Equals("hn"))
                    {
                        hn = resultsArray[j].AsEnumerable<float>().ToArray();
                    }
                    if (resultsArray[j].Name.Equals("cn"))
                    {
                        cn = resultsArray[j].AsEnumerable<float>().ToArray();
                    }
                }
                if (hn != null && cn != null)
                {
                    modelOutput.ModelOutStates.Add(hn);
                    modelOutput.ModelOutStates.Add(cn);
                }
            }
            catch (Exception ex)
            {
                throw new Exception("ModelProj failed", ex);
            }
            return modelOutput;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_modelSession != null)
                {
                    _modelSession.Dispose();
                }
                if (_customMetadata != null)
                {
                    _customMetadata = null;
                }
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
