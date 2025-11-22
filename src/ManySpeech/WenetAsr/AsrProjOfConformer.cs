using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Linq;
using ManySpeech.WenetAsr.Model;
using ManySpeech.WenetAsr.Utils;

namespace ManySpeech.WenetAsr
{
    internal class AsrProjOfConformer : IAsrProj,IDisposable
    {
        private bool _disposed;

        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _ctcSession;
        private CustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _unk_id = 1;
        private int _sos_eos_id = 0;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;
        public AsrProjOfConformer(AsrModel asrModel)
        {
            _encoderSession = asrModel.EncoderSession;
            _decoderSession = asrModel.DecoderSession;
            _ctcSession = asrModel.CtcSession;
            _blank_id = asrModel.Blank_id;
            _sos_eos_id = asrModel.Sos_eos_id;
            _unk_id = asrModel.Unk_id;
            _featureDim = asrModel.FeatureDim;
            _sampleRate = asrModel.SampleRate;
            _customMetadata = asrModel.CustomMetadata;
            _chunkLength = asrModel.ChunkLength;
            _shiftLength = asrModel.ShiftLength;
            _required_cache_size = asrModel.Required_cache_size;
        }
        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession CtcSession { get => _ctcSession; set => _ctcSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }

        public List<float[]> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> statesList = new List<float[]>();
            //计算尺寸
            int required_cache_size = _required_cache_size < 0 ? 0 : _required_cache_size;
            float[] att_cache = new float[_customMetadata.Num_blocks * _customMetadata.Head * required_cache_size * (_customMetadata.Output_size / _customMetadata.Head * 2)];
            float[] cnn_cache = new float[_customMetadata.Num_blocks * 1 * _customMetadata.Output_size * (_customMetadata.Cnn_module_kernel - 1)];
            statesList.Add(att_cache);
            statesList.Add(cnn_cache);
            return statesList;
        }
        private float[] InitCacheFeats(int batchSize = 1)
        {

            int cached_feature_size = 0;//1 + _right_context - _subsampling_rate;//TODO temp test
            float[] cacheFeats = new float[batchSize * cached_feature_size * 80];
            return cacheFeats;
        }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            states = statesList[0];
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 2 == 0, "when stack_states, state_list[0] is 2x");
            statesList.Add(states);
            return statesList;
        }
        public EncoderOutputEntity EncoderProj(List<AsrInputEntity> modelInputs, List<float[]> statesList, int offset)
        {
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "chunk")
                {
                    int[] dim = new int[] { 1, padSequence.Length / _customMetadata.Feature_size, _customMetadata.Feature_size };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "offset")
                {
                    int[] dim = new int[] { 1 };
                    Int64[] offset_tensor = new Int64[] { offset };
                    var tensor = new DenseTensor<Int64>(offset_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
                if (name == "required_cache_size")
                {
                    int required_cache_size = _required_cache_size;
                    int[] dim = new int[] { 1 };
                    Int64[] required_cache_size_tensor = new Int64[] { required_cache_size };
                    var tensor = new DenseTensor<Int64>(required_cache_size_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
                if (name == "att_cache")
                {
                    float[] att_cache = statesList[0];
                    int required_cache_size = att_cache.Length / _customMetadata.Num_blocks / _customMetadata.Head / (_customMetadata.Output_size / _customMetadata.Head * 2);
                    int[] dim = new int[] { _customMetadata.Num_blocks, _customMetadata.Head, required_cache_size, _customMetadata.Output_size / _customMetadata.Head * 2 };
                    var tensor = new DenseTensor<float>(att_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "cnn_cache")
                {
                    float[] cnn_cache = statesList[1];
                    int[] dim = new int[] { _customMetadata.Num_blocks, 1, _customMetadata.Output_size, _customMetadata.Cnn_module_kernel - 1 };
                    var tensor = new DenseTensor<float>(cnn_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "att_mask")
                {
                    int required_cache_size = _required_cache_size;
                    int[] dim = new int[] { required_cache_size + _customMetadata.Chunk_size, 1 };
                    bool[] att_mask = new bool[(required_cache_size + _customMetadata.Chunk_size) * 1];
                    att_mask = att_mask.Select(x => x = true).ToArray();
                    var tensor = new DenseTensor<bool>(att_mask, dim, false);
                    if (_customMetadata.Left_chunks > 0)
                    {
                        dim = new int[] { 1, 1, required_cache_size + _customMetadata.Chunk_size };
                        att_mask = new bool[1 * 1 * (required_cache_size + _customMetadata.Chunk_size)];
                        att_mask = att_mask.Select(x => x = true).ToArray();
                        int chunk_idx = offset / _customMetadata.Chunk_size - _customMetadata.Left_chunks;
                        if (chunk_idx < _customMetadata.Left_chunks)
                        {
                            for (int i = 0; i < (_customMetadata.Left_chunks - chunk_idx) * _customMetadata.Chunk_size; ++i)
                            {
                                att_mask[i] = false;
                            }
                        }
                        tensor = new DenseTensor<bool>(att_mask, dim, false);
                    }
                    container.Add(NamedOnnxValue.CreateFromTensor<bool>(name, tensor));
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
                    encoderOutput.Index = outputTensor.Dimensions[1];
                    encoderOutput.Output = outputTensor.ToArray();
                    var rAttCacheTensor = encoderResultsArray[1].AsTensor<float>();
                    encoderOutput.R_att_cache = rAttCacheTensor.ToArray();
                    var rCnnCache = encoderResultsArray[2].AsTensor<float>();
                    encoderOutput.R_cnn_cache = rCnnCache.ToArray();
                    List<float[]> next_statesList = new List<float[]>();
                    foreach (var item in encoderResultsArray)
                    {
                        if (item.Name.EndsWith("_cache"))
                        {
                            next_statesList.Add(item.AsEnumerable<float>().ToArray());
                        }
                    }
                    encoderOutput.StatesList = next_statesList;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return encoderOutput;
        }

        public CtcOutputEntity CtcProj(EncoderOutputEntity encoderOutput)
        {
            CustomMetadata customMetadata = _customMetadata;
            float[] padSequence = encoderOutput.Output;
            var inputMeta = _ctcSession.InputMetadata;
            CtcOutputEntity ctcOutput = new CtcOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hidden")
                {
                    int[] dim = new int[] { 1, padSequence.Length / customMetadata.Output_size, customMetadata.Output_size };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> ctcResults = null;
                ctcResults = _ctcSession.Run(container);

                if (ctcResults != null)
                {
                    var ctcResultsArray = ctcResults.ToArray();
                    var probsTensor = ctcResultsArray[0].AsTensor<float>();
                    ctcOutput.Probs = probsTensor.ToArray();

                    List<Int64[]> token_nums = new List<Int64[]> { };
                    List<Int64> token_len = new List<Int64>();

                    for (int i = 0; i < probsTensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[probsTensor.Dimensions[1]];
                        for (int j = 0; j < probsTensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < probsTensor.Dimensions[2]; k++)
                            {
                                token_num = probsTensor[i, j, token_num] > probsTensor[i, j, k] ? token_num : k;
                            }
                            item[j] = token_num;
                        }
                        List<Int64> item1 = new List<Int64>(item);
                        item1.Remove(item1.First());
                        List<Int64> item2 = new List<Int64>(item);
                        item2.RemoveAt(item2.Count - 1);
                        List<Int64> newItem = new List<Int64>();
                        int itemIndex = 0;
#if NET6_0_OR_GREATER
            // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
                        foreach (var itemTemp in item1.Zip<Int64, Int64>(item2))
                        {
                            if (itemTemp.First != itemTemp.Second)
                            {
                                newItem.Add(item[itemIndex]);
                            }
                            itemIndex++;
                        }
#else
                        // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
                        for (int n = 0; i < item1.Count && n < item2.Count; n++)
                        {
                            if (item1[n] != item2[n])
                            {
                                newItem.Add(item[n]);
                            }
                        }
#endif
                        newItem.Add(item.Last());
                        item = newItem.ToArray();
                        token_nums.Add(item);
                        token_len.Add(item.Length);

                    }
                    ctcOutput.Hyps = token_nums;
                    ctcOutput.Hyps_lens = token_len;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return ctcOutput;
        }

        public DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, CtcOutputEntity ctcOutputEntity, int batchSize = 1)
        {
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hyps")
                {
                    int[] dim = new int[] { batchSize, ctcOutputEntity.Hyps[0].Length };
                    var tensor = new DenseTensor<Int64>(ctcOutputEntity.Hyps[0], dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "hyps_lens")
                {
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<Int64>(ctcOutputEntity.Hyps_lens.ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "encoder_out")
                {
                    int[] dim = new int[] { batchSize, encoderOutputEntity.Output.Length / customMetadata.Output_size / batchSize, customMetadata.Output_size };
                    var tensor = new DenseTensor<float>(encoderOutputEntity.Output, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                List<float> rescoring_score = new List<float>();

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> score_tensor = decoderResultsArray[0].AsTensor<float>();
                    //method 1
                    List<float> scoreList = new List<float>();
                    for (int i = 0; i < score_tensor.Dimensions[0]; i++)
                    {
                        float score = 0.0f;
                        Int64[] item = new Int64[score_tensor.Dimensions[1]];
                        for (int j = 0; j < score_tensor.Dimensions[1]; j++)
                        {
                            float currScore = score_tensor[i, j, (int)ctcOutputEntity.Hyps[i][j]];
                            score += currScore;
                        }
                        scoreList.Add(score);
                    }
                    Tensor<float> r_score_tensor = decoderResultsArray[1].AsTensor<float>();
                    List<float> r_scoreList = new List<float>();
                    for (int i = 0; i < r_score_tensor.Dimensions[0]; i++)
                    {
                        float r_score = 0.0f;
                        if (_customMetadata.Is_bidirectional_decoder && _customMetadata.Reverse_weight > 0)
                        {
                            Int64[] item = new Int64[r_score_tensor.Dimensions[1]];
                            for (int j = 0; j < r_score_tensor.Dimensions[1]; j++)
                            {
                                float currScore = r_score_tensor[i, j, (int)ctcOutputEntity.Hyps[i][j]];
                                r_score += currScore;
                            }
                        }
                        r_scoreList.Add(r_score);
                    }
                    DecodeOptionsEntity decodeOptions = new DecodeOptionsEntity();
#if NET6_0_OR_GREATER
                    // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
                    foreach(var item in scoreList.Zip<float, float>(r_scoreList))
                    {
                        float score = item.First;
                        float r_score = item.Second;
#else
                    // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
                    for (int i = 0; i < scoreList.Count && i < r_scoreList.Count; i++)
                    {
                        float score = scoreList[i];
                        float r_score = r_scoreList[i];
#endif
                        float reverse_weight = _customMetadata.Reverse_weight;
                        // combined left-to-right and right-to-left score
                        float rescoring_score_item = score * (1 - reverse_weight) + r_score * reverse_weight;
                        rescoring_score.Add(rescoring_score_item);
                    }
                    decoderOutputEntity.Rescoring_score = rescoring_score;
                    //method 2
                    int num_hyps = ctcOutputEntity.Hyps.Count;
                    int max_hyps_len = 0;
                    int decode_out_len = score_tensor.Dimensions[2];
                    for (int i = 0; i < num_hyps; ++i)
                    {
                        int length = ctcOutputEntity.Hyps[i].Length;// + 1
                        max_hyps_len = Math.Max(length, max_hyps_len);
                        //hyps_lens.emplace_back(static_cast<int64_t>(length));
                    }
                    float[] rescoring_score2 = new float[ctcOutputEntity.Hyps.Count];
                    for (int i = 0; i < num_hyps; i++)
                    {
                        Int64[] hyp = ctcOutputEntity.Hyps[0];
                        float score = 0.0f;
                        // left to right decoder score
                        score = ComputeAttentionScore(
                            score_tensor.Skip(max_hyps_len * decode_out_len * i).ToArray(), hyp, _customMetadata.Eos_symbol, decode_out_len);
                        // Optional: Used for right to left score
                        float r_score = 0.0f;
                        if (_customMetadata.Is_bidirectional_decoder && _customMetadata.Reverse_weight > 0)
                        {
                            Int64[] r_hyp = new List<Int64>(hyp).ToArray();
                            // right to left decoder score
                            r_score = ComputeAttentionScore(
                                r_score_tensor.Skip(max_hyps_len * decode_out_len * i).ToArray(), r_hyp, _customMetadata.Eos_symbol, decode_out_len);
                        }
                        // combined left-to-right and right-to-left score
                        rescoring_score2[i] =
                            score * (1 - _customMetadata.Reverse_weight) + r_score * _customMetadata.Reverse_weight;
                    }

                }
            }
            catch (Exception ex)
            {
                //
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
                    if (_ctcSession != null)
                    {
                        _ctcSession = null;
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
        ~AsrProjOfConformer()
        {
            Dispose(_disposed);
        }
    }
}
