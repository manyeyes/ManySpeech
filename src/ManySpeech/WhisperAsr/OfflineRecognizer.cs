// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private Tokenizer _tokenizer;
        private IOfflineProj? _offlineProj;
        private OfflineModel _offlineModel;
        private bool _isDetectLanguage = false;
        private DecodingOptions? _decodingOptions;
        private ModelDimensions? _modelDimensions;

        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, ConfEntity? confEntity = null, string configFilePath = "", int threadsNum = 1)
        {
            if (confEntity == null)
            {
                confEntity = new ConfEntity();
                if (!string.IsNullOrEmpty(configFilePath))
                {
                    if (configFilePath.ToLower().EndsWith(".json"))
                    {
                        confEntity = LoadJsonConf(configFilePath);
                    }
                    else if (configFilePath.ToLower().EndsWith(".yaml"))
                    {
                        confEntity = LoadYamlConf(configFilePath);
                    }
                }
            }
            _offlineModel = new OfflineModel(encoderFilePath, decoderFilePath, confEntity: confEntity ?? new ConfEntity(), threadsNum: threadsNum);
            _decodingOptions = confEntity?.decoding_options;
            _modelDimensions = confEntity?.model_dimensions;
            if (_decodingOptions == null)
            {
                _decodingOptions = new DecodingOptions();
            }
            if (!confEntity?.is_multilingual ?? false)
            {
                _isDetectLanguage = false;
                _decodingOptions.language = null;// only "en"
                _offlineModel.SuppressSample = float.NaN; 
            }
            else
            {
                _isDetectLanguage = string.IsNullOrEmpty(_decodingOptions.language) ? true : false;
                _offlineModel.SuppressSample = float.NegativeInfinity;
            }
            _tokenizer = GetTokenizer(multilingual: confEntity?.is_multilingual ?? false, language: _decodingOptions.language, task: _decodingOptions.task, numLanguages: confEntity?.num_languages ?? 99);
            _offlineProj = new OfflineProj(_offlineModel);
        }
        private ConfEntity? LoadJsonConf(string configFilePath)
        {
            if (string.IsNullOrWhiteSpace(configFilePath))
            {
                return null;
            }
            ConfEntity? confEntity = Utils.PreloadHelper.ReadJson(configFilePath);
            return confEntity;
        }
        private ConfEntity? LoadYamlConf(string configFilePath)
        {
            if (string.IsNullOrWhiteSpace(configFilePath))
            {
                return null;
            }
            ConfEntity? confEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
            return confEntity;
        }

        private Tokenizer GetTokenizer(bool multilingual, string? language = null, string? task = null, int numLanguages = 99)
        {
            if (!string.IsNullOrEmpty(language))
            {
                language = language.ToLower();
                if (!Tiktoken.Dict.LANGUAGES.ContainsKey(language))
                {
                    if (Tiktoken.Dict.TO_LANGUAGE_CODE.ContainsKey(language))
                    {
                        language = Tiktoken.Dict.TO_LANGUAGE_CODE.GetValueOrDefault(language);
                    }
                    else
                    {
                        Console.WriteLine(string.Format("\"Unsupported language: {0}", language));
                        throw new Exception(string.Format("\"Unsupported language: {0}", language));
                    }
                }
            }
            string encodingName = string.Empty;
            if (multilingual)
            {
                encodingName = "multilingual";
                language = language ?? "en";
                task = task ?? "transcribe";
            }
            else
            {
                encodingName = "gpt2";
                language = null;
                task = null;
            }
            Tiktoken.GptEncoding gptEncoding = Tiktoken.GptEncoding.GetEncoding(encodingName: encodingName, numLanguages: numLanguages);
            return new Tokenizer(gptEncoding, language: language, task: task);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream offlineStream = new OfflineStream(_offlineModel);
            return offlineStream;
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
            //this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private DetectLanguageEntity DetectLanguage(EncoderOutputEntity encoderOutputEntity)
        {
            DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
            DecoderOutputEntity decoderOutputEntityForDetectLanguage = new DecoderOutputEntity();
            decoderOutputEntityForDetectLanguage = _offlineProj.DetectLanguage(encoderOutputEntity, _tokenizer.Sot);
            Tensor<float>? logits_tensor = decoderOutputEntityForDetectLanguage.Logits;
            if (logits_tensor != null)
            {
                // collect detected languages; suppress all non-language tokens
                bool[] mask = new bool[logits_tensor.Dimensions[2]];
                mask = mask.Select(x => x = true).ToArray();
                foreach (int token in _tokenizer.AllLanguageTokens)
                {
                    mask[token] = false;
                }
                List<int[]> languageTokens = new List<int[]> { };

                for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                {
                    int[] item = new int[logits_tensor.Dimensions[1]];
                    for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                    {
                        int token = 0;
                        for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                        {
                            if (!mask[k] || !mask[0])
                            {
                                token = logits_tensor[i, j, token] > logits_tensor[i, j, k] ? token : k;
                            }
                        }
                        item[j] = (int)token;
                    }
                    languageTokens.Add(item);
                }
                // decode languageTokens
                foreach (int[] languageToken in languageTokens)
                {
                    detectLanguageEntity.LanguageCodes.Add(_tokenizer.AllLanguageCodes[_tokenizer.AllLanguageTokens.IndexOf(languageToken[0])]);
                }
            }
            return detectLanguageEntity;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<List<Int64>> tokens = new List<List<Int64>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                OfflineInputEntity offlineInputEntity = new OfflineInputEntity();

                offlineInputEntity.Speech = stream.GetAllDecodeChunk();
                offlineInputEntity.SampleLength = stream.RealSampleLen;
                if (offlineInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                offlineInputEntity.SpeechLength = offlineInputEntity.Speech.Length;
                modelInputs.Add(offlineInputEntity);
                statesList.Add(stream.States);
                tokens.Add(stream.Tokens.Select(x=>(Int64)x).ToList());
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
                // 计算 mel->audio_features
                EncoderOutputEntity encoderOutputEntity = _offlineProj.EncoderProj(modelInputs);
                // 检测语种
                DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
                if (_isDetectLanguage)
                {
                    detectLanguageEntity = DetectLanguage(encoderOutputEntity);
                }
                // InitTokens
                string language = _decodingOptions.language;
                List<List<Int64>> initialTokensList = new List<List<long>>();
                for (int i = 0; i < batchSize; i++)
                {
                    if (detectLanguageEntity.LanguageCodes.Count == batchSize)
                    {
                        language = detectLanguageEntity.LanguageCodes[i];
                    }
                    List<Int64> initialTokens = GetInitTokens(language: language).ToList();
                    initialTokensList.Add(initialTokens);
                }
                int sample_begin = initialTokensList[0].Count;
                // logit_filters
                List<ILogitFilter> logitFilters = new List<ILogitFilter>();
                if (_decodingOptions.suppress_blank)
                {
                    logitFilters.Add(new SuppressBlank(_tokenizer, sample_begin));
                }
                if (_decodingOptions.suppress_tokens.Length > 0)
                {
                    int[] suppressTokens = GetSuppressTokens();
                    logitFilters.Add(new SuppressTokens(suppressTokens));
                }
                if (!_decodingOptions.without_timestamps)
                {
                    float precision = (float)_offlineProj.ChunkLength / _modelDimensions.n_audio_ctx;
                    int? maxInitialTimestampIndex = null;
                    if (_decodingOptions.max_initial_timestamp != null)
                    {
                        maxInitialTimestampIndex = (int)Math.Round((decimal)(_decodingOptions.max_initial_timestamp / precision));
                    }
                    logitFilters.Add(new ApplyTimestampRules(_tokenizer, sample_begin, maxInitialTimestampIndex));
                }
                // 识别
                float[] sumLogprobs = new float[batchSize];
                float[] noSpeechProbs = new float[batchSize];
                int sampleLen = 0;
                if (_modelDimensions != null)
                {
                    sampleLen = _decodingOptions.sample_len ?? _modelDimensions.n_text_ctx / 2;
                }
                for (int t = 0; t < sampleLen; t++)
                {
                    List<List<Int64>> tokensParams = new List<List<Int64>>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        if (tokens[i].Count > 0 && tokens[i].Count < initialTokensList[i].Count)
                        {
                            List<Int64> lastTokens = new List<Int64>();
                            lastTokens.Add(tokens[i].Last());
                            tokensParams.Add(lastTokens);
                        }
                        else
                        {
                            if (tokens[i].Count == 0)
                            {
                                tokens[i] = new List<Int64>(initialTokensList[i]);
                            }
                            tokensParams.Add(tokens[i]);
                        }
                    }
                    DecoderOutputEntity decoderOutputEntity = _offlineProj.DecoderProj(encoderOutputEntity, tokensParams);
                    Tensor<float>? logits_tensor = decoderOutputEntity.Logits;
                    if (t == 0 && _tokenizer.NoSpeech != int.MinValue)
                    {

                        List<float[]> probsAtSot = new List<float[]>();
                        //List<int[]> languageTokens = new List<int[]> { };
                        for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                        {
                            int sotIndex = initialTokensList[i].IndexOf(_tokenizer.Sot);
                            for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                            {
                                if (j == sotIndex)
                                {
                                    float[] item = new float[logits_tensor.Dimensions[2]];
                                    for (int k = 0; k < logits_tensor.Dimensions[2]; k++)
                                    {
                                        item[k] = (int)logits_tensor[i, j, k];
                                    }

                                    probsAtSot.Add(item);
                                }
                            }
                        }
                        probsAtSot = probsAtSot.Select(x => x = ComputeHelper.SoftmaxCompute(x)).ToList();
                        for (int i = 0; i < probsAtSot.Count; i++)
                        {
                            noSpeechProbs[i] = probsAtSot[i][_tokenizer.NoSpeech];
                        }

                    }
                    List<float[]> logits_tensor_last = new List<float[]>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            float[] item = new float[logits_tensor.Dimensions[2]];
                            if (j == logits_tensor.Dimensions[1] - 1)
                            {
                                for (int k = 0; k < logits_tensor.Dimensions[2]; k++)
                                {
                                    item[k] = logits_tensor[i, j, k];
                                }
                                logits_tensor_last.Add(item);
                            }
                        }
                    }
                    // logit_filters
                    foreach (ILogitFilter logitFilter in logitFilters)
                    {
                        List<List<float>> logitsTensorLastList = logits_tensor_last.Select(x => x.ToList()).ToList();
                        logitFilter.apply(ref logitsTensorLastList, tokens);
                        logits_tensor_last = logitsTensorLastList.Select(x => x.ToArray()).ToList();
                    }
                    // expand the tokens tensor with the selected next tokens
                    bool completed = false;
                    tokens = GreedyDecoder(tokens, logits_tensor_last, ref sumLogprobs, out completed); // GreadySearch
                    //if completed or tokens.shape[-1] > self.n_ctx:
                    if (completed || tokens[0].Count > _modelDimensions?.n_text_ctx)
                    {
                        break;
                    }
                }
                //计算tokens
                // make sure each sequence has at least one EOT token at the end
                // tokens = F.pad(tokens, (0, 1), value = self.eot)
                foreach (List<Int64> token in tokens)
                {
                    token.Add(_tokenizer.Eot);
                }
                List<List<Int64>> newTokens = new List<List<long>>();
                foreach (List<Int64> token in tokens)
                {
                    List<Int64> newToken = new List<Int64>();
                    for (int i = 0; i < token.Count; i++)
                    {
                        if (i >= sample_begin)
                        {
                            if (token.IndexOf(_tokenizer.Eot) != i)
                            {
                                if (token[i] != _tokenizer.Eot)
                                {
                                    newToken.Add(token[i]);
                                }
                            }
                        }
                    }
                    newTokens.Add(newToken);
                }
                // into stream
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.Tokens = newTokens[streamIndex].Select(x=>(int)x).ToList();
                    stream.RemoveAllDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private int[] GetSuppressTokens()
        {
            int[] suppressTokens = _decodingOptions.suppress_tokens;
            string suppressTokensStr = _decodingOptions.suppress_tokens_str;
            if (!string.IsNullOrEmpty(suppressTokensStr))
            {
                suppressTokens = suppressTokensStr.Split(',').Select(x => (int)x.ToCharArray()[0]).ToArray();//TODO 这里或许需要约定suppressTokensStr的形式：是以“,”逗号隔开的字符，还是一个多个字符组成的字符串？
            }
            List<int> suppressTokenList = new List<int>();
            if (suppressTokens.Contains(-1))
            {
                suppressTokens = suppressTokens.Where(x => x >= 0).ToArray();
                suppressTokenList = suppressTokens.ToList();
                suppressTokenList.AddRange(_tokenizer.NonSpeechTokens);
                suppressTokens = suppressTokenList.ToArray();
                suppressTokenList = new List<int>();
            }
            else if (suppressTokens == null || suppressTokens.Length == 0)
            {
                suppressTokens = new int[0];
            }
            suppressTokenList = suppressTokens.ToList();
            suppressTokenList.Add(_tokenizer.Transcribe);
            suppressTokenList.Add(_tokenizer.Translate);
            suppressTokenList.Add(_tokenizer.Sot);
            suppressTokenList.Add(_tokenizer.SotPrev);
            suppressTokenList.Add(_tokenizer.SotLm);
            if (_tokenizer.NoSpeech != null && _tokenizer.NoSpeech != int.MinValue)
            {
                suppressTokenList.Add(_tokenizer.NoSpeech);
            }
            suppressTokenList.Sort();
            suppressTokens = suppressTokenList.ToArray();
            suppressTokenList = new List<int>();
            return suppressTokens;
        }

        private List<List<Int64>> GreedyDecoder(List<List<Int64>> tokens, List<float[]> logits_tensor_last, ref float[] sumLogprobs, out bool completed)
        {
            float temperature = _decodingOptions.temperature;
            int eot = _tokenizer.Eot;
            List<List<Int64>> nextTokens = new List<List<Int64>>();
            if (temperature == 0)
            {
                for (int j = 0; j < logits_tensor_last.Count; j++)
                {
                    Int64[] item = new Int64[1];
                    int token = 0;
                    for (int k = 1; k < logits_tensor_last[j].Length; k++)
                    {
                        token = logits_tensor_last[j][token] > logits_tensor_last[j][k] ? token : k;
                    }
                    item[0] = (int)token;
                    nextTokens.Add(item.ToList());
                }
            }
            else
            {
                for (int j = 0; j < logits_tensor_last.Count; j++)
                {
                    Int64[] item = new Int64[1];
                    int token = SimpleCategorical.Sample(logits_tensor_last[j], temperature);
                    item[0] = (int)token;
                    nextTokens.Add(item.ToList());
                }
            }
            List<float[]> logprobs = logits_tensor_last.Select(x => x = ComputeHelper.LogCompute(ComputeHelper.SoftmaxCompute(x))).ToList();
            List<float[]> currentLogprobs = new List<float[]>() { new float[] { logits_tensor_last[0][nextTokens[0][0]] } };
            sumLogprobs = sumLogprobs.Select(x => x = x + currentLogprobs[0][0]).ToArray();
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Last() == eot)
                {
                    nextTokens[i][nextTokens[i].Count - 1] = eot;
                }
            }
            for (int i = 0; i < tokens.Count; i++)
            {
                tokens[i].AddRange(nextTokens[i]);
            }
            completed = false;
            foreach (List<Int64> longs in nextTokens)
            {
                if (longs.Last() != eot)
                {
                    completed = false;
                    break;
                }
                else
                {
                    completed = true;
                }
            }
            string text_result = _tokenizer.decode(tokens[0].Select(x => (int)x).ToList());
            System.Diagnostics.Debug.WriteLine(text_result);
            return tokens;
        }
        public Int64[] GetInitTokens(string language = "")
        {
            if (!string.IsNullOrEmpty(language))
            {
                _tokenizer.ChangeSotSequence(language);
            }
            List<int> sotSequence = _tokenizer.SotSequence;
            if (_decodingOptions.without_timestamps)
            {
                sotSequence = _tokenizer.SotSequenceIncludingNotimestamps;
            }
            List<int> tokens = new List<int>();
            //ConfEntity? confEntity = _offlineProj?.ConfEntity;
            if (_modelDimensions != null)
            {
                List<int> prompt_tokens = new List<int>();
                string prompt_str = _decodingOptions?.prompt_str ?? string.Empty;
                if (!string.IsNullOrEmpty(prompt_str))
                {
                    prompt_tokens = _tokenizer.encode(" " + prompt_str.Trim());
                }
                else
                {
                    List<int>? prompt = _decodingOptions?.prompt ?? null;
                    if (prompt != null)
                    {
                        prompt_tokens = prompt;
                    }
                }
                //tokens=new List<int>();
                if (prompt_tokens.Count > 0)
                {
                    int? max_prompt_len = _modelDimensions?.n_text_ctx / 2 - 1;
                    for (int i = 0; i < prompt_tokens.Count; i++)
                    {
                        if (i < prompt_tokens.Count - max_prompt_len)
                        {
                            prompt_tokens.RemoveAt(i);
                        }
                    }
                    tokens.Add(_tokenizer.SotPrev);
                    tokens.AddRange(prompt_tokens);
                }
                tokens.AddRange(sotSequence);
                List<int> prefix_tokens = new List<int>();
                string prefix_str = _decodingOptions?.prefix_str ?? string.Empty;
                if (!string.IsNullOrEmpty(prefix_str))
                {
                    prefix_tokens = _tokenizer.encode(" " + prefix_str.Trim());
                }
                else
                {
                    List<int>? prefix = _decodingOptions?.prefix ?? null;
                    if (prefix != null)
                    {
                        prefix_tokens = prefix;
                    }
                }
                if (prefix_tokens.Count > 0)
                {
                    if (_decodingOptions?.sample_len != null &&
                        _decodingOptions?.sample_len != int.MinValue)
                    {
                        int? max_prefix_len = _modelDimensions?.n_text_ctx / 2 - _decodingOptions?.sample_len;

                        for (int i = 0; i < prefix_tokens.Count; i++)
                        {
                            if (i < prefix_tokens.Count - max_prefix_len)
                            {
                                prefix_tokens.RemoveAt(i);
                            }
                        }
                    }
                    tokens.AddRange(prefix_tokens);
                }
            }
            return tokens.Select(x => (Int64)x).ToArray();
        }
        // TODO:MaximumLikelihoodRanker
        private void MaximumLikelihoodRanker()
        {

        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                List<int> token_num = stream.Tokens;
                //string text_result = "";                
                string text_result = string.Empty;
                if (!_decodingOptions.without_timestamps)
                {
                    text_result = _tokenizer.DecodeWithTimestamps(token_num.Select(x => (int)x).ToList());
                }
                else
                {
                    text_result = _tokenizer.decode(token_num.Select(x => (int)x).ToList());
                }
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                offlineRecognizerResultEntity.Text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", " ").Replace("▁", "").ToLower();
                offlineRecognizerResultEntity.Tokens = stream.Tokens;
                //offlineRecognizerResultEntity.Timestamps = stream.Timestamps;
                //offlineRecognizerResultEntity.Language = stream.Language;
                offlineRecognizerResultEntities.Add(offlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return offlineRecognizerResultEntities;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_offlineProj != null)
                    {
                        _offlineProj.Dispose();
                    }
                    if (_offlineModel != null)
                    {
                        _offlineModel.Dispose();
                    }
                    if (_tokenizer != null)
                    {
                        _tokenizer.Dispose();
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
        ~OfflineRecognizer()
        {
            Dispose(_disposed);
        }
    }
}