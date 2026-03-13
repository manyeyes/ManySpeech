using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using ManySpeech.AudioTagging.Model;
using ManySpeech.AudioTagging.Utils;

namespace ManySpeech.AudioTagging
{
    internal class OfflineProjOfCED : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;
        public OfflineProjOfCED(OfflineModel offlineModel)
        {
            _modelSession = offlineModel.ModelSession;
            _featureDim = offlineModel.FeatureDim;
            _sampleRate = offlineModel.SampleRate;
            _chunkLength = offlineModel.ChunkLength;
            _shiftLength = offlineModel.ShiftLength;
            _required_cache_size = offlineModel.Required_cache_size;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }

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
                int statesItemTemp_item_axisnum = 1280;
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
            int batchSize = states[0].Length / 1280;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 1280;
                    int statesItemTemp_size = 1 * 1280;
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

        public ModelOutputEntity ModelProj(List<ModelInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / _featureDim).ToArray();
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _modelSession.InputMetadata;
            ModelOutputEntity modelOutputEntity = new ModelOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "mel_spec")
                {
                    int[] dim = new int[] { batchSize, _featureDim, padSequence.Length / _featureDim / batchSize };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> modelResults = null;
                modelResults = _modelSession.Run(container);

                if (modelResults != null)
                {
                    var modelResultsArray = modelResults.ToArray();
                    var outputTensor = modelResultsArray[0].AsTensor<float>();
                    modelOutputEntity.ModelOut = outputTensor;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("ModelProj failed", ex);
            }
            return modelOutputEntity;
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
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~OfflineProjOfCED()
        {
            Dispose(_disposed);
        }
    }
}
