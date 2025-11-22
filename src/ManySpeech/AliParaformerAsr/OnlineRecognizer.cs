// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace ManySpeech.AliParaformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OnlineRecognizer : IDisposable
    {
        private bool _disposed;

        private OnlineModel _onlineModel;
        private string[] _tokens;
        private List<float[]> _next_statesList=new List<float[]>();

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, string configFilePath, string mvnFilePath, string tokensFilePath, int threadsNum = 1)
        {
            _onlineModel = new OnlineModel(encoderFilePath, decoderFilePath, mvnFilePath, configFilePath, threadsNum);
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
        }
        public OnlineStream CreateOnlineStream()
        {
            OnlineStream onlineStream = new OnlineStream(_onlineModel);
            return onlineStream;
        }
        public OnlineRecognizerResultEntity GetResult(OnlineStream stream)
        {
            List<OnlineStream> streams = new List<OnlineStream>();
            streams.Add(stream);
            OnlineRecognizerResultEntity onlineRecognizerResultEntity = GetResults(streams)[0];

            return onlineRecognizerResultEntity;
        }

        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineStream> streams)
        {
            //this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = this.DecodeMulti(streams);

            return onlineRecognizerResultEntities;
        }

        private EncoderOutputEntity EncoderProj(List<OnlineInputEntity> modelInputs, int batchSize)
        {
            float[] padSequence = PadSequence_unittest(modelInputs);
            var inputMeta = _onlineModel.EncoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
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
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _onlineModel.EncoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    encoderOutput.Enc = new List<List<float[]>>();
                    var encTensor = encoderResultsArray[0].AsTensor<float>();
                    List<List<float[]>> enc = new List<List<float[]>>();
                    for (int i = 0; i < encTensor.Dimensions[0]; i++)
                    {
                        List<float[]> itemList = new List<float[]>();
                        for (int j = 0; j < encTensor.Dimensions[1]; j++)
                        {
                            float[] item = new float[encTensor.Dimensions[2]];
                            for (int k = 0; k < encTensor.Dimensions[2]; k++)
                            {
                                item[k] = encTensor[i, j, k];
                            }
                            itemList.Add(item);
                        }
                        enc.Add(itemList);
                    }
                    encoderOutput.Enc = enc;
                    encoderOutput.Enc_len = encoderResultsArray[1].AsEnumerable<Int32>().ToArray();
                    var alphasTensor = encoderResultsArray[2].AsTensor<float>();
                    List<List<float>> alphas = new List<List<float>>();
                    for (int i = 0; i < alphasTensor.Dimensions[0]; i++)
                    {
                        List<float> item = new List<float>();
                        for (int j = 0; j < alphasTensor.Dimensions[1]; j++)
                        {
                            item.Add(alphasTensor[i, j]);
                        }
                        alphas.Add(item);
                    }
                    encoderOutput.Alphas = alphas;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("EncoderProj failed", ex.InnerException);
            }
            return encoderOutput;
        }

        private PredictorOutputEntity PredictorProj(EncoderOutputEntity encoderOutput, List<OnlineStream> streams)
        {
            int batchSize = streams.Count;
            List<List<float>> newAlphas = new List<List<float>>();
            newAlphas = _onlineModel.DynamicMask(encoderOutput.Alphas);
            encoderOutput.Alphas = newAlphas;
            PredictorOutputEntity predictorOutputEntity = new PredictorOutputEntity();
            List<List<float[]>> cacheHiddens = new List<List<float[]>>();
            List<List<float>> cacheAlpha = new List<List<float>>();
            for (int i = 0; i < streams.Count; i++)
            {
                streams[i].CifHidden.AddRange(encoderOutput.Enc[i]);
                streams[i].CifAlpha.AddRange(encoderOutput.Alphas[i]);
                cacheHiddens.Add(streams[i].CifHidden);
                cacheAlpha.Add(streams[i].CifAlpha);
            }
            List<int> token_length = new List<int>();
            List<float[]> list_fires = new List<float[]>();
            List<List<float[]>> list_frames = new List<List<float[]>>();
            List<float> cache_alphas = new List<float>();
            List<float[]> cache_hiddens = new List<float[]>();

            int hidden_size = cacheHiddens[0][0].Length;
            int len_time = cacheAlpha[0].Count;
            for (int b = 0; b < batchSize; b++)
            {
                float integrate = 0.0F;
                float[] frames = new float[hidden_size];
                List<float[]> list_frame = new List<float[]>();
                List<float> list_fire = new List<float>();
                for (int j = 0; j < len_time; j++)
                {
                    //////////////////
                    float alpha = cacheAlpha[b][j];
                    //////////////////
                    float[] hiddens_item = new float[hidden_size];
                    hiddens_item = cacheHiddens[b][j];
                    //////////////////
                    if (alpha + integrate < _onlineModel.ConfEntity.predictor_conf.threshold)
                    {
                        integrate += alpha;
                        list_fire.Add(integrate);
                        float[] hiddens_item_temp = hiddens_item.Select(x => alpha * x).ToArray();
                        for (int framesIndex = 0; framesIndex < frames.Length; framesIndex++)
                        {
                            frames[framesIndex] += hiddens_item_temp[framesIndex];
                        }
                    }
                    else
                    {
                        float[] hiddens_item_temp = hiddens_item.Select(x => (_onlineModel.ConfEntity.predictor_conf.threshold - integrate) * x).ToArray();
                        for (int framesIndex = 0; framesIndex < frames.Length; framesIndex++)
                        {
                            frames[framesIndex] += hiddens_item_temp[framesIndex];
                        }
                        list_frame.Add(frames);
                        integrate += alpha;
                        list_fire.Add(integrate);
                        integrate -= _onlineModel.ConfEntity.predictor_conf.threshold;
                        frames = hiddens_item.Select(x => integrate * x).ToArray();
                    }
                }
                cache_alphas.Add(integrate);
                if (integrate > 0.0F)
                {
                    float[] cache_hiddens_item = frames.Select(x => x / integrate).ToArray();
                    cache_hiddens.Add(cache_hiddens_item);
                }
                else
                {
                    cache_hiddens.Add(frames);
                }
                token_length.Add(list_frame.Count);
                list_fires.Add(list_fire.ToArray());
                list_frames.Add(list_frame);
            }
            int max_token_len = token_length.Max(x => x);
            List<float[]> list_ls = new List<float[]>();
            for (int b = 0; b < batchSize; b++)
            {
                float[] pad_frames = new float[(max_token_len - token_length[b]) * hidden_size];
                if (token_length[b] == 0)
                {
                    list_ls.Add(pad_frames);
                }
                else
                {
                    list_frames[b].Add(pad_frames);
                    list_ls.AddRange(list_frames[b]);
                }
            }
            float[] cache_alphas_arr = cache_alphas.ToArray();

            for (int b = 0; b < batchSize; b++)
            {
                streams[b].CifAlpha = new List<float>();
                streams[b].CifAlpha.Add(cache_alphas_arr[b]);
                streams[b].CifHidden = new List<float[]>();
                streams[b].CifHidden.Add(cache_hiddens[b]);
            }
            List<float> ls_stack = new List<float>();
            for (int list_ls_index = 0; list_ls_index < list_ls.Count; list_ls_index++)
            {
                ls_stack.AddRange(list_ls[list_ls_index]);
            }
            float[] list_ls_stack = ls_stack.ToArray();
            predictorOutputEntity.Acoustic_embeds = list_ls_stack;
            predictorOutputEntity.Acoustic_embeds_len = token_length.ToArray();
            return predictorOutputEntity;
        }

        private DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, PredictorOutputEntity predictorOutputEntity, List<float[]> stackStatesList, int batchSize)
        {
            float[] enc = new float[encoderOutputEntity.Enc.Count * encoderOutputEntity.Enc[0].Count * encoderOutputEntity.Enc[0][0].Length];
            batchSize = encoderOutputEntity.Enc.Count;
            int d = 0;
            for (int b = 0; b < batchSize; b++)
            {
                foreach (float[] item in encoderOutputEntity.Enc[b])
                {
                    Array.Copy(item, 0, enc, d* item.Length, item.Length);
                    d++;
                }
            }
            int[] enc_len = encoderOutputEntity.Enc_len;
            float[] acoustic_embeds = predictorOutputEntity.Acoustic_embeds;
            int[] acoustic_embeds_len = predictorOutputEntity.Acoustic_embeds_len;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _onlineModel.DecoderSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "enc")
                {
                    int[] dim = new int[] { batchSize, enc.Length / 512 / batchSize, 512 };//inputMeta[name].Dimensions
                    var tensor = new DenseTensor<float>(enc, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "enc_len")
                {
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(enc_len, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name == "acoustic_embeds")
                {
                    int[] dim = new int[] { batchSize, acoustic_embeds.Length / 512 / batchSize, 512 };
                    var tensor = new DenseTensor<float>(acoustic_embeds, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "acoustic_embeds_len")
                {
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<int>(acoustic_embeds_len, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<int>(name, tensor));
                }
                if (name.StartsWith("in_cache_"))
                {
                    for (int i = 0; i < stackStatesList.Count; i++)
                    {
                        float[] in_cache_item = stackStatesList[i];
                        if (name == "in_cache_" + i.ToString())
                        {
                            int[] dim = new int[] { batchSize, 512, 10 };
                            var tensor = new DenseTensor<float>(in_cache_item, dim, false);
                            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                        }
                    }
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
                    decoderOutputEntity.Logits = logits_tensor.ToArray();
                    decoderOutputEntity.Sample_ids = token_nums;
                    List<float[]> statesList = new List<float[]>();
                    foreach (var item in decoderResultsArray)
                    {
                        if (item.Name.StartsWith("out_cache_"))
                        {
                            statesList.Add(item.AsEnumerable<float>().ToArray());
                        }
                    }
                    decoderOutputEntity.StatesList = statesList;

                }
            }
            catch (Exception ex)
            {
                throw new Exception("DecoderProj failed", ex);
            }
            return decoderOutputEntity;
        }

        private void Forward(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OnlineStream> streamsWorking = new List<OnlineStream>();
            List<OnlineInputEntity> modelInputs = new List<OnlineInputEntity>();
            List<List<float[]>> stateList = new List<List<float[]>>();
            List<Int64[]> hypList = new List<Int64[]>();
            List<List<Int64>> tokens = new List<List<Int64>>();
            int padFrameNum = _onlineModel.ChunkLength;
            int shiftFrameNum = _onlineModel.ShiftLength;
            foreach (OnlineStream stream in streams)
            {
                OnlineInputEntity onlineInputEntity = new OnlineInputEntity();
                onlineInputEntity.Speech = stream.GetDecodeChunk(padFrameNum);
                if (onlineInputEntity.Speech == null)
                {
                    continue;
                }
                onlineInputEntity.SpeechLength = onlineInputEntity.Speech.Length;
                modelInputs.Add(onlineInputEntity);
                hypList.Add(stream.Hyp);
                stateList.Add(stream.States);
                tokens.Add(stream.Tokens);
                streamsWorking.Add(stream);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            try
            {
                int batchSize = modelInputs.Count;
                List<float[]> states = new List<float[]>();
                List<float[]> stackStatesList = new List<float[]>();
                stackStatesList = _onlineModel.stack_states(stateList);
                EncoderOutputEntity encoderOutput = EncoderProj(modelInputs, batchSize);
                PredictorOutputEntity predictorConfEntity = PredictorProj(encoderOutput, streamsWorking);
                if (predictorConfEntity.Acoustic_embeds.Length > 0)
                {
                    DecoderOutputEntity decoderOutputEntity = DecoderProj(encoderOutput, predictorConfEntity, stackStatesList, batchSize);
                    List<List<float[]>> next_statesList = new List<List<float[]>>();
                    next_statesList = _onlineModel.unstack_states(decoderOutputEntity.StatesList);
                    _next_statesList = decoderOutputEntity.StatesList;
                    int streamIndex = 0;
                    foreach (OnlineStream stream in streamsWorking)
                    {
                        stream.Tokens.AddRange(decoderOutputEntity.Sample_ids[streamIndex].ToList());
                        stream.States = next_statesList[streamIndex];
                        streamIndex++;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Online recognition failed", ex);
            }

        }

        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OnlineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (_tokens[token] != "</s>" && _tokens[token] != "<s>" && _tokens[token] != "<blank>" && _tokens[token] != "<unk>")
                    {
                        if (IsChinese(_tokens[token], true))
                        {
                            text_result += _tokens[token];
                        }
                        else
                        {
                            text_result += "▁" + _tokens[token] + "▁";
                        }
                    }
                }
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                onlineRecognizerResultEntity.Text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁", " ").Replace("▁", "").ToLower();
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

        private float[] PadSequence_unittest(List<OnlineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);// + 80 * _addPadSequence  560*30  + 560 * 19
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? curr_speech = modelInputs[i].Speech;
                Array.Copy(curr_speech, 0, speech, i * curr_speech.Length, curr_speech.Length);
            }
            speech = speech.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
            return speech;
        }

        private float[] PadSequence(List<OnlineInputEntity> modelInputs)
        {
            int max_speech_length = modelInputs.Max(x => x.SpeechLength);//+560*19
            int speech_length = max_speech_length * modelInputs.Count;
            float[] speech = new float[speech_length];
            float[,] xxx = new float[modelInputs.Count, max_speech_length];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                if (max_speech_length == modelInputs[i].SpeechLength)
                {
                    for (int j = 0; j < xxx.GetLength(1); j++)
                    {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                        xxx[i, j] = modelInputs[i].Speech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。
                    }
                    continue;
                }
                float[] nullspeech = new float[max_speech_length - modelInputs[i].SpeechLength];
                float[]? curr_speech = modelInputs[i].Speech;
                float[] padspeech = new float[max_speech_length];
                Array.Copy(curr_speech, 0, padspeech, 0, curr_speech.Length);
                for (int j = 0; j < padspeech.Length; j++)
                {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                    xxx[i, j] = padspeech[j];
#pragma warning restore CS8602 // 解引用可能出现空引用。 
                }

            }
            int s = 0;
            for (int i = 0; i < xxx.GetLength(0); i++)
            {
                for (int j = 0; j < xxx.GetLength(1); j++)
                {
                    speech[s] = xxx[i, j];
                    s++;
                }
            }
            return speech;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onlineModel != null)
                    {
                        _onlineModel.Dispose();
                    }
                    if (_tokens != null)
                    {
                        _tokens = null;
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
        ~OnlineRecognizer()
        {
            Dispose(_disposed);
        }
    }
}