// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManySpeech.WenetAsr2.Model;
using ManySpeech.WenetAsr2.Utils;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;
using System.IO;

namespace ManySpeech.WenetAsr2
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizer
    {
        private OfflineModel _onlineModel;
        private readonly ILogger<OfflineRecognizer> _logger;
        private string[] _tokens;

        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string ctcFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 1)
        {
            _onlineModel = new OfflineModel(encoderFilePath, decoderFilePath, ctcFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);

            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OfflineRecognizer>(loggerFactory);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_onlineModel);
            return onlineStream;
        }
        public OfflineRecognizerResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            OfflineRecognizerResultEntity offlineRecognizerResultEntity = GetResults(streams)[0];
            return offlineRecognizerResultEntity;
        }
        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }
        #region proj
        private EncoderOutputEntity EncoderProj(List<OfflineInputEntity> modelInputs, List<float[]> statesList, int offset)
        {
            CustomMetadata customMetadata = _onlineModel.CustomMetadata;
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _onlineModel.EncoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "chunk")
                {
                    int[] dim = new int[] { 1, padSequence.Length / customMetadata.Feature_size, customMetadata.Feature_size };
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
                    int required_cache_size = _onlineModel.Required_cache_size;
                    int[] dim = new int[] { 1 };
                    Int64[] required_cache_size_tensor = new Int64[] { required_cache_size };
                    var tensor = new DenseTensor<Int64>(required_cache_size_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
                if (name == "att_cache")
                {
                    float[] att_cache = statesList[0];
                    int required_cache_size =  att_cache.Length / customMetadata.Num_blocks / customMetadata.Head / (customMetadata.Output_size / customMetadata.Head * 2);
                    int[] dim = new int[] { customMetadata.Num_blocks, customMetadata.Head, required_cache_size, customMetadata.Output_size / customMetadata.Head * 2 };
                    var tensor = new DenseTensor<float>(att_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "cnn_cache")
                {
                    float[] cnn_cache = statesList[1];
                    int[] dim = new int[] { customMetadata.Num_blocks, 1, customMetadata.Output_size, customMetadata.Cnn_module_kernel - 1 };
                    var tensor = new DenseTensor<float>(cnn_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "att_mask")
                {
                    int required_cache_size = _onlineModel.Required_cache_size;
                    int[] dim = new int[] { required_cache_size + customMetadata.Chunk_size, 1 };
                    bool[] att_mask = new bool[(required_cache_size + customMetadata.Chunk_size) * 1]; 
                    att_mask= att_mask.Select(x=>x=true).ToArray();
                    var tensor = new DenseTensor<bool>(att_mask, dim, false);
                    if (customMetadata.Left_chunks > 0)
                    {
                        dim = new int[] { 1, 1, required_cache_size + customMetadata.Chunk_size };
                        att_mask = new bool[1 * 1 * (required_cache_size + customMetadata.Chunk_size)];
                        att_mask = att_mask.Select(x => x = true).ToArray();
                        int chunk_idx = offset / customMetadata.Chunk_size - customMetadata.Left_chunks;
                        if (chunk_idx < customMetadata.Left_chunks)
                        {
                            for (int i = 0; i < (customMetadata.Left_chunks - chunk_idx) * customMetadata.Chunk_size; ++i)
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
                encoderResults = _onlineModel.EncoderSession.Run(container);

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

        private CtcOutputEntity CtcProj(EncoderOutputEntity encoderOutput)
        {
            CustomMetadata customMetadata = _onlineModel.CustomMetadata;
            float[] padSequence = encoderOutput.Output;
            var inputMeta = _onlineModel.CtcSession.InputMetadata;
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
                ctcResults = _onlineModel.CtcSession.Run(container);

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
        #endregion proj

        private DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, CtcOutputEntity ctcOutputEntity, int batchSize = 1)
        {
            CustomMetadata customMetadata = _onlineModel.CustomMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _onlineModel.DecoderSession.InputMetadata;
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
                decoderResults = _onlineModel.DecoderSession.Run(container);

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logits_tensor = decoderResultsArray[0].AsTensor<float>();
                    List<Int64[]> token_nums = new List<Int64[]> { };

                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[logits_tensor.Dimensions[1]];
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

                    Tensor<float> r_score_tensor = decoderResultsArray[1].AsTensor<float>();
                    List<Int64[]> r_token_nums = new List<Int64[]> { };

                    for (int i = 0; i < r_score_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[r_score_tensor.Dimensions[1]];
                        for (int j = 0; j < r_score_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < r_score_tensor.Dimensions[2]; k++)
                            {
                                token_num = r_score_tensor[i, j, token_num] > r_score_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                        }
                        r_token_nums.Add(item);
                    }
                    decoderOutputEntity.Logits = logits_tensor.ToArray();
                    decoderOutputEntity.Sample_ids = r_token_nums;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return decoderOutputEntity;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            int contextSize = 2;//_onlineModel.CustomMetadata.Context_size;
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<Int64[]> hypList = new List<Int64[]>();
            //List<Int64>[] tokens = new List<Int64>[batchSize];
            //Int64[] hyps = new Int64[_context_size * batchSize];
            List<List<Int64>> tokens = new List<List<Int64>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                OfflineInputEntity onlineInputEntity = new OfflineInputEntity();

                onlineInputEntity.Speech = stream.GetDecodeChunk();
                if (onlineInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                onlineInputEntity.SpeechLength = onlineInputEntity.Speech.Length;
                modelInputs.Add(onlineInputEntity);
                hypList.Add(stream.Hyp);
                statesList.Add(stream.States);
                tokens.Add(stream.Tokens);
                streamsWorking.Add(stream);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OfflineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                int batchSize = modelInputs.Count;
                int offset = streams[0].Offset;
                List<float[]> stackStatesList = new List<float[]>();
                stackStatesList = _onlineModel.stack_states(statesList);
                EncoderOutputEntity encoderOutputEntity = EncoderProj(modelInputs, stackStatesList, offset);
                CtcOutputEntity ctcOutputEntity = CtcProj(encoderOutputEntity);
                DecoderOutputEntity decoderOutputEntity = DecoderProj(encoderOutputEntity, ctcOutputEntity);
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                next_statesList = _onlineModel.unstack_states(encoderOutputEntity.StatesList);
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.Tokens.AddRange(ctcOutputEntity.Hyps[streamIndex].ToList());
                    stream.States = next_statesList[streamIndex];
                    stream.Offset = offset + encoderOutputEntity.Index;
                    stream.RemoveDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }

        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (_tokens[token].Split(' ')[0] != "</s>" && _tokens[token].Split(' ')[0] != "<s>" && _tokens[token].Split(' ')[0] != "<sos/eos>" && _tokens[token].Split(' ')[0] != "<blank>" && _tokens[token].Split(' ')[0] != "<unk>")
                    {
                        if (IsChinese(_tokens[token].Split(' ')[0], true))
                        {
                            text_result += _tokens[token].Split(' ')[0];
                        }
                        else
                        {
                            text_result += "▁" + _tokens[token].Split(' ')[0] + "▁";
                        }
                    }
                }
                OfflineRecognizerResultEntity onlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                onlineRecognizerResultEntity.Text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", " ").Replace("▁", "").ToLower();
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return onlineRecognizerResultEntities;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        private bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }
    }
}