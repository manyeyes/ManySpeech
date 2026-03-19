using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using ManySpeech.DolphinAsr.Model;
using ManySpeech.DolphinAsr.Utils;
using System;

namespace ManySpeech.DolphinAsr
{
    internal class OfflineProjOfDolphin : IOfflineProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private CustomMetadata _customMetadata;
        private OfflineModel _offlineModel;
        public OfflineProjOfDolphin(OfflineModel offlineModel)
        {
            _encoderSession = offlineModel.EncoderSession;
            _decoderSession = offlineModel.DecoderSession;
            _customMetadata = offlineModel.CustomMetadata;
            _offlineModel = offlineModel;
        }
        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            int batchSize = statesList.Count;
            Debug.Assert(statesList[0].Count % 6 == 0, "when stack_states, state_list[0] is 6x");
            int fsmnLayer = statesList[0].Count;
            for (int i = 0; i < fsmnLayer; i++)
            {
                float[] statesItemTemp = new float[statesList[0][i].Length * batchSize];
                int statesItemTemp_item_length = statesList[0][i].Length;
                int statesItemTemp_item_axisnum = 512;
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
            Debug.Assert(states.Count % 6 == 0, "when stack_states, state_list[0] is 6x");
            int fsmnLayer = states.Count;
            int batchSize = states[0].Length / 512;
            for (int b = 0; b < batchSize; b++)
            {
                List<float[]> statesListItem = new List<float[]>();
                for (int j = 0; j < fsmnLayer; j++)
                {
                    float[] item = states[j];
                    int statesItemTemp_axisnum = 512;
                    int statesItemTemp_size = 1 * 512;
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

        public EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength).ToArray();
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "speech")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / batchSize };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "speech_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    Int64[] input_lengths_tensor = new Int64[batchSize];
                    input_lengths_tensor = inputLengths;
                    var tensor = new DenseTensor<Int64>(input_lengths_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    var outputTensor = encoderResultsArray[0].AsTensor<float>();
                    encoderOutput.Output = outputTensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }

        public DecoderOutputEntity DecoderProj(List<List<int>> tokenidsList, float[] encoder_outputs)
        {
            List<int[]> ys = new List<int[]>();
            ys = tokenidsList.Select(x => x.ToArray()).ToList();
            //Console.WriteLine(string.Join(",",ys[0]));
            int batchSize = ys.Count;
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            try
            {
                foreach (var name in inputMeta.Keys)
                {
                    if (name == "ys")
                    {
                        int[] dim = new int[2] { batchSize, ys[0].Length };
                        var tensor = new DenseTensor<Int64>(ys.SelectMany(x => x.Select(x=>(Int64)x)).ToArray(), dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                    }
                    if (name == "enc_out")
                    {
                        int[] dim = new int[3] { batchSize, encoder_outputs.Length / _offlineModel.ConfEntity.encoder_conf.output_size / batchSize, _offlineModel.ConfEntity.encoder_conf.output_size };
                        var tensor = new DenseTensor<float>(encoder_outputs, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                List<float> rescoring_score = new List<float>();
                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logits_tensor = decoderResultsArray[0].AsTensor<float>();
                    decoderOutputEntity.LogitsTensor = logits_tensor;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("DecoderProj failed", ex);
            }
            return decoderOutputEntity;
        }
        public List<List<int>> DecodeAsr(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        public List<List<int>> DetectLanguage(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if(j<_offlineModel.FirstLangId || j > _offlineModel.LastLangId)
                    {
                        continue;
                    }
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        public List<List<int>> DetectRegion(Tensor<float> logitsTensor)
        {
            int batchSize = logitsTensor.Dimensions[0];
            int numClasses = logitsTensor.Dimensions[1];
            var tokenIdsList = new List<List<int>>(batchSize);

            for (int i = 0; i < batchSize; i++)
            {
                int bestIndex = 0; // 初始假设索引 0 最大
                for (int j = 1; j < numClasses; j++)
                {
                    if (j < _offlineModel.FirstRegionId || j > _offlineModel.LastRegionId)
                    {
                        continue;
                    }
                    if (logitsTensor[i, j] > logitsTensor[i, bestIndex])
                    {
                        bestIndex = j;
                    }
                }
                tokenIdsList.Add(new List<int> { bestIndex });
            }
            return tokenIdsList;
        }

        private float ComputeAttentionScore(float[] prob, Int64[] hyp, int eos, int decode_out_len)
        {
            float score = 0.0f;
            for (int j = 0; j < hyp.Length; j++)
            {
                score += prob[j * decode_out_len + hyp[j]];
            }
            //score += prob[hyp.Length * decode_out_len + eos];
            return score;
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
                }
                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        ~OfflineProjOfDolphin()
        {
            Dispose(_disposed);
        }
    }
}
