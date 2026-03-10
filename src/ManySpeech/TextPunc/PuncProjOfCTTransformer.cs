// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.TextPunc.Model;
using ManySpeech.TextPunc.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.TextPunc
{
    internal class PuncProjOfCTTransformer : IPuncProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;
        //private int _blank_id = 0;
        //private int _sos_eos_id = 1;
        //private int _unk_id = 2;

        //private int _featureDim = 80;
        //private int _sampleRate = 16000;

        public PuncProjOfCTTransformer(PuncModel puncModel)
        {
            _modelSession = puncModel.ModelSession;
            //_blank_id = puncModel.Blank_id;
            //_sos_eos_id = puncModel.Sos_eos_id;
            //_unk_id = puncModel.Unk_id;
            //_featureDim = puncModel.FeatureDim;
            //_sampleRate = puncModel.SampleRate;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        //public int Blank_id { get => _blank_id; set => _blank_id = value; }
        //public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        //public int Unk_id { get => _unk_id; set => _unk_id = value; }
        //public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        //public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        
        public PuncOutputEntity ModelProj(PuncInputEntity modelInput)
        {
            int BatchSize = 1;
            PuncOutputEntity modelOutput = new PuncOutputEntity();
            try
            {
                var inputMeta = _modelSession.InputMetadata;
                var container = new List<NamedOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "inputs")
                    {
                        int[] dim = new int[] { BatchSize, modelInput.TextLengths / 1 / BatchSize };
                        var tensor = new DenseTensor<int>(modelInput.MiniSentenceId, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                    }
                    if (name == "text_lengths")
                    {
                        int[] dim = new int[] { BatchSize };
                        int[] text_lengths = new int[BatchSize];
                        for (int i = 0; i < BatchSize; i++)
                        {
                            text_lengths[i] = modelInput.TextLengths / 1 / BatchSize;
                        }
                        var tensor = new DenseTensor<int>(text_lengths, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                    }
                }
                IReadOnlyCollection<string> outputNames = new List<string>();
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = null;

                results = _modelSession.Run(container);
                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    modelOutput.Logits = resultsArray[0].AsEnumerable<float>().ToArray();
                    Tensor<float> logits_tensor = resultsArray[0].AsTensor<float>();
                    List<int[]> token_nums = new List<int[]> { };

                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        int[] item = new int[logits_tensor.Dimensions[1]];
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                        }
                        token_nums.Add(item);
                    }
                    modelOutput.Punctuations = token_nums;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Automatic punctuation failed", ex);
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
        ~PuncProjOfCTTransformer()
        {
            Dispose(_disposed);
        }
    }
}
