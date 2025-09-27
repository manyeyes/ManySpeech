// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using ManySpeech.K2TransducerAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.K2TransducerAsr
{
    internal class OfflineProjOfZipformer2ctc : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _joinerSession;
        private OfflineCustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;

        public OfflineProjOfZipformer2ctc(OfflineModel offlineModel)
        {
            _encoderSession = offlineModel.EncoderSession;
            _decoderSession = offlineModel.DecoderSession;
            _joinerSession = offlineModel.JoinerSession;
            _blank_id = offlineModel.Blank_id;
            _sos_eos_id = offlineModel.Sos_eos_id;
            _unk_id = offlineModel.Unk_id;
            _featureDim = offlineModel.FeatureDim;

            _customMetadata = new OfflineCustomMetadata();
            _customMetadata = offlineModel.CustomMetadata;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession JoinerSession { get => _joinerSession; set => _joinerSession = value; }
        public OfflineCustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }

        public EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs, int batchSize)
        {
            //int featureDim = _featureDim;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / FeatureDim / batchSize, FeatureDim };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "x_lens")
                {
                    int[] dim = new int[] { batchSize };
                    Int64[] speech_lengths = new Int64[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        speech_lengths[i] = padSequence.Length / FeatureDim / batchSize;
                    }
                    var tensor = new DenseTensor<Int64>(speech_lengths, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.encoder_out = encoderResultsArray[0].AsEnumerable<float>().ToArray();
                    encoderOutput.encoder_out_lens = encoderResultsArray[1].AsEnumerable<Int64>().ToArray();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }
        public DecoderOutputEntity DecoderProj(Int64[]? decoder_input, int batchSize)
        {            
            return null;
        }

        public JoinerOutputEntity JoinerProj(float[]? encoder_out, float[]? decoder_out)
        {
            return null;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
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
                    if (_joinerSession != null)
                    {
                        _joinerSession.Dispose();
                    }
                    if (_customMetadata != null)
                    {
                        _customMetadata = null;
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
        ~OfflineProjOfZipformer2ctc()
        {
            Dispose(_disposed);
        }
    }
}
