using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using ManySpeech.FireRedAsr.Model;
using ManySpeech.FireRedAsr.Utils;
using System;

namespace ManySpeech.FireRedAsr
{
    internal class AsrProjOfAED : IAsrProj, IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private CustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _pad_id = 2;
        private int _sos_id = 3;
        private int _eos_id = 4;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;
        public AsrProjOfAED(AsrModel asrModel)
        {
            _encoderSession = asrModel.EncoderSession;
            _decoderSession = asrModel.DecoderSession;
            _blank_id = asrModel.Blank_id;
            _unk_id = asrModel.Unk_id;
            _pad_id = asrModel.Pad_id;
            _sos_id = asrModel.Sos_id;
            _eos_id = asrModel.Eos_id;
            _featureDim = asrModel.FeatureDim;
            _sampleRate = asrModel.SampleRate;
            _customMetadata = asrModel.CustomMetadata;
            _chunkLength = asrModel.ChunkLength;
            _shiftLength = asrModel.ShiftLength;
            _required_cache_size = asrModel.Required_cache_size;
        }
        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int Pad_id { get => _pad_id; set => _pad_id = value; }
        public int Sos_id { get => _sos_id; set => _sos_id = value; }
        public int Eos_id { get => _eos_id; set => _eos_id = value; }
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

        public EncoderOutputEntity EncoderProj(List<AsrInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            Int64[] inputLengths = modelInputs.Select(x => (long)x.SpeechLength / _featureDim).ToArray();
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "input")
                {
                    int[] dim = new int[] { batchSize, padSequence.Length / _featureDim/ batchSize, _featureDim };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "input_lengths")
                {
                    int[] dim = new int[] { batchSize };
                    Int64[] input_lengths_tensor = new Int64[batchSize];
                    input_lengths_tensor= inputLengths;
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
                    var outputLengthsTensor = encoderResultsArray[1].AsTensor<Int64>();
                    var maskTensor = encoderResultsArray[2].AsTensor<bool>();
                    encoderOutput.Output = outputTensor.ToArray();
                    encoderOutput.OutputLengths = outputLengthsTensor.ToArray();
                    encoderOutput.Mask = maskTensor.ToArray();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex);
            }
            return encoderOutput;
        }

        public DecoderOutputEntity DecoderProj(List<List<Int64>> tokensList, float[] encoder_outputs, bool[] src_mask, List<float[]> cacheList)
        {
            List<Int64[]> ys = new List<Int64[]>();
            ys = tokensList.Select(x => x.ToArray()).ToList();
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
                        var tensor = new DenseTensor<Int64>(ys.SelectMany(x => x).ToArray(), dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                    }
                    if (name == "encoder_outputs")
                    {
                        int[] dim = new int[3] { batchSize, encoder_outputs.Length / 1280/ batchSize, 1280 };
                        var tensor = new DenseTensor<float>(encoder_outputs, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }
                    if (name == "src_mask")
                    {
                        int[] dim = new int[3] { batchSize, 1, src_mask.Length / 1/ batchSize };
                        var tensor = new DenseTensor<bool>(src_mask, dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<bool>(name, tensor));
                    }
                    for(int i = 0; i < 16; i++)
                    {
                        if (name == "cache_"+i.ToString())
                        {
                            int[] dim = new int[3] { batchSize, cacheList[i].Length / 1280/ batchSize, 1280 };
                            var tensor = new DenseTensor<float>(cacheList[i], dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }
                    }
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                List<float> rescoring_score = new List<float>();
                //计算ys和cacheList
                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logits_tensor = decoderResultsArray[0].AsTensor<float>();
                    List<float[]> logits_tensor_list = new List<float[]>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        float[] row = new float[logits_tensor.Dimensions[1]];
                        for (int j = 0; j < row.Length; j++)
                        {
                            row[j] = logits_tensor[i, j];
                        }
                        logits_tensor_list.Add(row);
                    }
                    logits_tensor_list = logits_tensor_list.Select(x => x = ComputeHelper.LogCompute(ComputeHelper.SoftmaxCompute(x.Select(y => y / 1.25f).ToArray()))).ToList();
                    int[] item = new int[logits_tensor.Dimensions[0]];
                    for (int j = 0; j < logits_tensor_list.Count; j++)
                    {
                        int token_num = 0;
                        for (int k = 1; k < logits_tensor_list[j].Length; k++)
                        {
                            token_num = logits_tensor_list[j][token_num] > logits_tensor_list[j][k] ? token_num : k;
                        }
                        item[j] = (int)token_num;
                        //timestamps.Add(new int[] { 0, 0 });
                    }
                    cacheList = new List<float[]>();
                    foreach (var cache in decoderResultsArray.Skip(1))
                    {
                        cacheList.Add(cache.AsTensor<float>().ToArray());
                    }
                    decoderOutputEntity.TokensList = item.Select(x => new List<Int64> { (Int64)x }).ToList(); ;
                    decoderOutputEntity.CacheList = cacheList;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("DecoderProj failed", ex);
            }
            return decoderOutputEntity;
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
        ~AsrProjOfAED()
        {
            Dispose(_disposed);
        }
    }
}
