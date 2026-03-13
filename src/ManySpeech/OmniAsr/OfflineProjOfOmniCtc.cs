// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.OmniAsr.Model;
using ManySpeech.OmniAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.OmniAsr
{
    internal class OfflineProjOfOmniCtc : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _sampleRate = 16000;
        private int _featureDim = 1;

        public OfflineProjOfOmniCtc(OfflineModel offlineModel)
        {
            _modelSession = offlineModel.ModelSession;
            _sampleRate=offlineModel.SampleRate;
            _featureDim = offlineModel.FeatureDim;
        }

        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public OfflineOutputEntity ModelProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize=modelInputs.Count;
            int featureDim = _featureDim;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _modelSession.InputMetadata;
            OfflineOutputEntity modelOutput = new OfflineOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "seqs")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / batchSize, featureDim };
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
                    modelOutput.Logits = modelResultsArray[0].AsTensor<float>();
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
        ~OfflineProjOfOmniCtc()
        {
            Dispose(_disposed);
        }
    }
}
