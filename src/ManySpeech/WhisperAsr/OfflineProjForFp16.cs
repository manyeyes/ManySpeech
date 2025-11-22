// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.WhisperAsr
{
    internal class OfflineProjForFp16 : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private CustomMetadata _customMetadata;
        //private ConfEntity? _confEntity;

        private int _featureDim = 80;
        private int _sampleRate = 16000;

        private int _chunkLength = 0;
        private int _frameLength = 0;
        private int _shiftLength = 0;
        private int _hopLength = 0;

        public OfflineProjForFp16(OfflineModel offlineModel)
        {
            _encoderSession = offlineModel.EncoderSession;
            _decoderSession = offlineModel.DecoderSession;
            _featureDim = offlineModel.FeatureDim;
            _sampleRate = offlineModel.SampleRate;
            _chunkLength = offlineModel.ChunkLength;
            _frameLength = offlineModel.FrameLength;
            _shiftLength = offlineModel.ShiftLength;
            _hopLength = offlineModel.HopLength;

            _customMetadata = new CustomMetadata();
            _customMetadata = offlineModel.CustomMetadata;
            //_confEntity = offlineModel.ConfEntity;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        //public ConfEntity ConfEntity { get => _confEntity; set => _confEntity = value; }
        public int FrameLength { get => _frameLength; set => _frameLength = value; }
        public int HopLength { get => _hopLength; set => _hopLength = value; }

        public EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            float[] padSequence = PadHelper.PadSequence(modelInputs, tailLen: 0);
            Float16[] padSequence2 = FloatArrayConverter.FloatArrayToHalfArray(padSequence);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / 3000 / batchSize, 3000 };
                    var tensor = new DenseTensor<Float16>(padSequence2, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Float16>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    var outputTensor = encoderResultsArray[0].AsTensor<Float16>();
                    encoderOutput.Dim = outputTensor.Dimensions.ToArray();
                    //encoderOutput.Output = outputTensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return encoderOutput;
        }
        public DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, List<List<Int64>> tokens)
        {
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int nAudio = encoderOutputEntity.Dim[0];

                    int[] dim = new int[] { nAudio, tokens[0].Count };
                    List<Int64> longs = new List<Int64>();
                    for (int i = 0; i < Math.Min(nAudio, tokens.Count); i++)
                    {
                        longs.AddRange(tokens[i]);
                    }
                    var tensor = new DenseTensor<Int64>(longs.ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "xa")
                {


                    int[] dim = encoderOutputEntity.Dim;
                    var tensor = new DenseTensor<float>(encoderOutputEntity.Output, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logitsTensor = decoderResultsArray[0].AsTensor<float>();


                    decoderOutputEntity.Logits = logitsTensor;
                    decoderOutputEntity.Dim = logitsTensor.Dimensions.ToArray();

                }
            }
            catch (Exception ex)
            {
                //
            }
            return decoderOutputEntity;
        }

        public DecoderOutputEntity DetectLanguage(EncoderOutputEntity encoderOutputEntity, Int64 tokenizerSot)
        {
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "x")
                {
                    int nAudio = encoderOutputEntity.Dim[0];

                    int[] dim = new int[] { nAudio, 1 };
                    List<Int64> tokens = new List<Int64>();
                    Int64[] longItem = new Int64[] { tokenizerSot };
                    for (int i = 0; i < nAudio; i++)
                    {
                        tokens.AddRange(longItem);
                    }
                    var tensor = new DenseTensor<Int64>(tokens.ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "xa")
                {
                    int[] dim = encoderOutputEntity.Dim;
                    var tensor = new DenseTensor<float>(encoderOutputEntity.Output, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logitsTensor = decoderResultsArray[0].AsTensor<float>();
                    decoderOutputEntity.Logits = logitsTensor;
                    decoderOutputEntity.Dim = logitsTensor.Dimensions.ToArray();
                }
            }
            catch (Exception ex)
            {
                //
            }
            return decoderOutputEntity;
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
        ~OfflineProjForFp16()
        {
            Dispose(_disposed);
        }
    }
}
