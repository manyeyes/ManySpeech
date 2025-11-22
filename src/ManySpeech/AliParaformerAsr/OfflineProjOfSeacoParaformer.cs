// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.AliParaformerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliParaformerAsr
{
    internal class OfflineProjOfSeacoParaformer : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _modelSession;
        private EmbedSeacoModel _seacohwModel;
        private Tensor<float>? _hwEmbed = null;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _sampleRate = 16000;

        public OfflineProjOfSeacoParaformer(OfflineModel offlineModel)
        {
            _modelSession = offlineModel.ModelSession;
            var inputMeta = _modelSession.InputMetadata;
            if (inputMeta.ContainsKey("bias_embed"))
            {
                _seacohwModel = new EmbedSeacoModel(offlineModel.ModelebFilePath);
                List<int[]>? hotwords = offlineModel.Hotwords;
                _hwEmbed = _seacohwModel.Forward(hotwords);
            }
            _blank_id = offlineModel.Blank_id;
            _sos_eos_id = offlineModel.Sos_eos_id;
            _unk_id = offlineModel.Unk_id;
            _featureDim = offlineModel.FeatureDim;
            _sampleRate = offlineModel.SampleRate;
        }
        public InferenceSession ModelSession { get => _modelSession; set => _modelSession = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }

        public ModelOutputEntity ModelProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Tensor<float>? hwEmbed = null;
            List<int[]>? hotwords = modelInputs.SelectMany(x => x.Hotwords).ToList();
            if (hotwords != null && hotwords?.Count > 0)
            {
                hwEmbed = _seacohwModel.Forward(hotwords);
            }
            else
            {
                hwEmbed = _hwEmbed;
            }
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _modelSession.InputMetadata;
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 560 / batchSize, 560 };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    int[] speech_lengths = new int[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / 560 / batchSize;
                    }
                    var tensor = new DenseTensor<int>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "bias_embed")
                {
                    int[] dim = new int[] { batchSize, 0, 512 };
                    float[] biasEmbed = new float[0];
                    if (hwEmbed != null)
                    {
                        long _hwEmbedLength = hwEmbed.Length;
                        biasEmbed = new float[_hwEmbedLength * batchSize];
                        List<float[]> ebList = new List<float[]>();
                        for (int n = 0; n < hwEmbed.Dimensions[1]; n++)
                        {
                            float[] eb = new float[10 * 512];
                            for (int j = 0; j < hwEmbed.Dimensions[0]; j++)
                            {
                                int k = hwEmbed.Dimensions[2];
                                Array.Copy(hwEmbed.ToArray(), j * hwEmbed.Dimensions[1] * k + n * k, eb, j * k, k);
                            }
                            ebList.Add(eb);
                        }
                        float[] biasEmbedTemp = ebList.SelectMany(x => x).ToArray(); // hwEmbed.ToArray();// 
                        for (int i = 0; i < batchSize; i++)
                        {
                            Array.Copy(biasEmbedTemp, 0, biasEmbed, i * biasEmbedTemp.Length, biasEmbedTemp.Length);
                        }
                        dim = new int[] { batchSize, biasEmbed.Length / 512 / batchSize, 512 };
                    }
                    var tensor = new DenseTensor<float>(biasEmbed, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            ModelOutputEntity modelOutputEntity = new ModelOutputEntity();
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _modelSession.Run(container);

                if (results != null)
                {
                    var resultsArray = results.ToArray();
                    modelOutputEntity.model_out = resultsArray[0].AsTensor<float>();
                    modelOutputEntity.model_out_lens = resultsArray[1].AsEnumerable<int>().ToArray();
                    if (resultsArray.Length >= 4)
                    {
                        Tensor<float> cif_peak_tensor = resultsArray[3].AsTensor<float>();
                        modelOutputEntity.cif_peak_tensor = cif_peak_tensor;
                    }
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
                    if (_seacohwModel != null)
                    {
                        _seacohwModel.Dispose();
                    }
                    if (_hwEmbed != null)
                    {
                        _hwEmbed = null;
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
        ~OfflineProjOfSeacoParaformer()
        {
            Dispose(_disposed);
        }
    }
}
