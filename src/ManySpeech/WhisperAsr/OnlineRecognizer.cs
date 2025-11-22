// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    /// <summary>
    /// online recognizer package
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class OnlineRecognizer : IDisposable
    {
        private bool _disposed;

        private Tokenizer _tokenizer;
        private IOnlineProj? _onlineProj;
        private OnlineModel _onlineModel;
        private bool _isDetectLanguage = false;
        private DecodingOptions? _decodingOptions;
        private ModelDimensions? _modelDimensions;
        // init value for transcribe
        private bool _verbose = true;
        private float[] _temperature = new float[] { 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };
        private float? _compression_ratio_threshold = 2.4f;
        private float _logprob_threshold = -1.0f;
        private float? _no_speech_threshold = 0.6f;
        // 值为true时，将代入上一次的token，作为initToken
        private bool _condition_on_previous_text = false;
        // 存储上一次的token,作为init token
        private string? _initial_prompt;
        private bool _word_timestamps = false;
        private string _prepend_punctuations = "\"'“¿([{-";
        private string _append_punctuations = "\"'.。,，!！?？:：”)]}、";
        private List<float> _clip_timestamps = new List<float>() { 0f };
        private float _hallucination_silence_threshold;
        private string _punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、";
        //
        private LinkedList<List<List<Int64>>> _streamingTokensList = new LinkedList<List<List<Int64>>>();

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, ConfEntity? confEntity = null, string configFilePath = "", int threadsNum = 1)
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
            _onlineModel = new OnlineModel(encoderFilePath, decoderFilePath, confEntity: confEntity ?? new ConfEntity(), configFilePath: configFilePath, threadsNum: threadsNum);

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
                _onlineModel.SuppressSample = float.NaN;
            }
            else
            {
                _isDetectLanguage = string.IsNullOrEmpty(_decodingOptions.language) ? true : false;
                _onlineModel.SuppressSample = float.NegativeInfinity;
            }
            _tokenizer = GetTokenizer(multilingual: confEntity?.is_multilingual ?? false, language: _decodingOptions.language, task: _decodingOptions.task, numLanguages: confEntity?.num_languages ?? 99);
            _onlineProj = new OnlineProj(_onlineModel);
        }
        private ConfEntity? LoadJsonConf(string configFilePath)
        {
            if (string.IsNullOrWhiteSpace(configFilePath))
            {
                return null;
            }
            //ConfEntity? confEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
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

        public void InitParams(bool verbose = false, bool word_timestamps = false)
        {
            if (word_timestamps || _decodingOptions.task == "translate")
            {
                Console.WriteLine("Word-level timestamps on translations may not be reliable.");
            }
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

        private async Task AddStreamingTokens(List<List<Int64>> tokensList, int sampleBegin)
        {
            List<List<Int64>> newTokensList = new List<List<long>>();
            foreach (List<Int64> tokens in tokensList)
            {
                List<Int64> tempTokens = new List<Int64>(tokens);
                List<Int64> newTokens = new List<Int64>();
                for (int i = 0; i < tempTokens.Count; i++)
                {
                    if (i >= sampleBegin)
                    {
                        if (tempTokens.IndexOf(_tokenizer.Eot) != i)
                        {
                            if (tempTokens[i] != _tokenizer.Eot)
                            {
                                newTokens.Add(tempTokens[i]);
                            }
                        }
                    }
                }
                newTokensList.Add(newTokens);
            }
            _streamingTokensList.AddFirst(newTokensList);
        }
        public List<string> GetStreamingTexts()
        {
            List<List<Int64>> tokensList = new List<List<long>>();
            if (_streamingTokensList.Count > 0)
            {
                tokensList = _streamingTokensList.First();
                _streamingTokensList.RemoveFirst();
            }
            if (_streamingTokensList.Count > 10)
            {
                for (int i = 0; i < tokensList.Count - 10; i++)
                {
                    _streamingTokensList.RemoveFirst();
                }
            }
            List<string> streamingTexts = new List<string>();
            foreach (var tokens in tokensList)
            {
                string text = "";
                if (!_decodingOptions.without_timestamps)
                {
                    text = _tokenizer.DecodeWithTimestamps(tokens.Select(x => (int)x).ToList()).Replace("<|notimestamps|>", "");//.Replace("<|endoftext|>", "");
                }
                else
                {
                    text = _tokenizer.decode(tokens.Select(x => (int)x).ToList());//.Replace("<|notimestamps|>", "").Replace("<|endoftext|>", "");
                }
                streamingTexts.Add(text);
            }
            return streamingTexts;
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

        private DetectLanguageEntity DetectLanguage(EncoderOutputEntity encoderOutputEntity)
        {
            DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
            DecoderOutputEntity decoderOutputEntityForDetectLanguage = new DecoderOutputEntity();
            decoderOutputEntityForDetectLanguage = _onlineProj.DetectLanguage(encoderOutputEntity, _tokenizer.Sot);
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

        private DecodingResultEntity Decoding(EncoderOutputEntity encoderOutputEntity, List<List<Int64>> initialTokensList, int batchSize)
        {
            DecodingResultEntity decodingResultEntity = new DecodingResultEntity();
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
                float precision = (float)_onlineProj.ChunkLength / _modelDimensions.n_audio_ctx;
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
            List<List<Int64>> tokens = new List<List<Int64>>();
            for (int tt = 0; tt < sampleLen; tt++)
            {
                List<List<Int64>> tokensParams = new List<List<Int64>>();
                for (int i = 0; i < batchSize; i++)
                {
                    if (tokens.Count == 0)
                    {
                        tokens.Add(new List<long>());
                    }
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
                DecoderOutputEntity decoderOutputEntity = _onlineProj.DecoderProj(encoderOutputEntity, tokensParams);
                Tensor<float>? logits_tensor = decoderOutputEntity.Logits;
                if (tt == 0 && _tokenizer.NoSpeech != int.MinValue)
                {
                    List<float[]> probsAtSot = new List<float[]>();
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
                // GreadySearch
                tokens = GreedyDecoder(tokens, logits_tensor_last, ref sumLogprobs, out completed);
                Task task = Task.Run(async () =>
                {
                    await AddStreamingTokens(tokens, sample_begin);
                });
                //task.Start();
                // if completed or tokens.shape[-1] > self.n_ctx:
                if (completed || tokens[0].Count > _modelDimensions.n_text_ctx)
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

            decodingResultEntity.Tokens = newTokens[0].Select(x => (int)x).ToList();
            decodingResultEntity.NoSpeechProb = noSpeechProbs.First();
            decodingResultEntity.Temperature = _decodingOptions.temperature;
#if NET6_0_OR_GREATER
            decodingResultEntity.AvgLogprob = newTokens.Zip<List<Int64>, float>(sumLogprobs).Select(x => (float)x.Second / (x.First.Count + 1)).FirstOrDefault();
#else
        // .NET 6.0以下版本：使用非泛型Zip方法（返回Tuple）
        for (int i = 0; i < newTokens.Count; i++)
        {
            // 获取当前索引的元素，增加空值保护
            var iTokens = newTokens[i];
            float logprob = sumLogprobs[i];
            
            // 计算除数（避免null和除零）
            int tokenCount = iTokens?.Count ?? 0;
            int divisor = tokenCount + 1;
            divisor = divisor <= 0 ? 1 : divisor; // 确保除数至少为1
            
            // 计算平均值（只取第一个元素，与原逻辑一致）
            decodingResultEntity.AvgLogprob = logprob / divisor;
            break; // 只处理第一个元素，跳出循环
        }
#endif
            string text = _tokenizer.decode(newTokens[0].Select(x => (int)x).ToList());
            //byte[] bytes = Encoding.UTF8.GetBytes(text);
            decodingResultEntity.CompressionRatio = ComputeHelper.CompressionRatio(text);
            return decodingResultEntity;
        }

        public DecodingResultEntity DecodeWithFallback(EncoderOutputEntity encoderOutputEntity, List<List<Int64>> initialTokensList, int batchSize)
        {
            DecodingResultEntity decodingResultEntity = new DecodingResultEntity();
            foreach (float t in _temperature)
            {
                if (t > 0)
                {
                    _decodingOptions.beam_size = null;
                    _decodingOptions.patience = null;
                }
                else
                {
                    _decodingOptions.best_of = null;
                }
                _decodingOptions.temperature = t;

                decodingResultEntity = Decoding(encoderOutputEntity, initialTokensList, batchSize);
                bool needsFallback = false;
                // too repetitive
                if (_compression_ratio_threshold != null && decodingResultEntity.CompressionRatio > _compression_ratio_threshold)
                {
                    needsFallback = true;
                }
                // average log probability is too low
                if (_logprob_threshold != null && decodingResultEntity.AvgLogprob < _logprob_threshold)
                {
                    needsFallback = true;
                }
                // silence
                if (_no_speech_threshold != null && decodingResultEntity.NoSpeechProb > _no_speech_threshold)
                {
                    needsFallback = false;
                }
                if (!needsFallback)
                {
                    break;
                }
            }
            return decodingResultEntity;
        }

        private void Forward(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OnlineStream> streamsWorking = new List<OnlineStream>();
            List<int> seekList = new List<int>();
            //List<string> languages = new List<string>();
            List<string> previousLanguages = new List<string>();
            List<List<int>> all_tokens_list = new List<List<int>>();
            List<List<SegmentEntity>> all_segments_list = new List<List<SegmentEntity>>();
            List<OnlineInputEntity> modelInputs = new List<OnlineInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<List<int>> tokens = new List<List<int>>();
            List<OnlineStream> streamsTemp = new List<OnlineStream>();
            List<int> prompt_reset_since_list = new List<int>();
            List<List<int>> decodingPromptList = new List<List<int>>();
            List<int> initial_prompt_tokens = new List<int>();
            if (!string.IsNullOrEmpty(_initial_prompt))
            {
                initial_prompt_tokens = _tokenizer.encode(" " + _initial_prompt.Trim());
            }
            else
            {
                initial_prompt_tokens = new List<int>();
            }

            foreach (OnlineStream stream in streams)
            {
                OnlineInputEntity onlineInputEntity = new OnlineInputEntity();
                onlineInputEntity.Speech = stream.GetDecodeChunk();
                if (onlineInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                onlineInputEntity.SpeechLength = onlineInputEntity.Speech.Length;
                modelInputs.Add(onlineInputEntity);
                //statesList.Add(stream.States);
                tokens.Add(stream.Tokens);
                if (stream.AllSegments.Count > 0)
                {
                    previousLanguages.Add(stream.AllSegments.Last().Language);
                }
                else
                {
                    previousLanguages.Add("");
                }
                if (stream.Tokens.Count == 0)
                {
                    stream.Tokens.AddRange(initial_prompt_tokens);
                }
                prompt_reset_since_list.Add(stream.Prompt_reset_since);
                decodingPromptList.Add(stream.DecodingPrompt);
                seekList.Add(stream.Seek);
                all_tokens_list.Add(stream.Tokens);
                all_segments_list.Add(stream.AllSegments);
                streamsWorking.Add(stream);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                int batchSize = modelInputs.Count;
                // 计算 mel->audio_features
                EncoderOutputEntity encoderOutputEntity = _onlineProj.EncoderProj(modelInputs);
                // 检测语种
                DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
                if (_isDetectLanguage)
                {
                    detectLanguageEntity = DetectLanguage(encoderOutputEntity);
                }
                // InitTokens
                List<List<Int64>> initialTokensList = new List<List<long>>();
                List<string> currLanguages = new List<string>();
                for (int i = 0; i < batchSize; i++)
                {
                    if (decodingPromptList[i] == null)
                    {
                        _decodingOptions.prompt = new List<int>();
                        for (int m = 0; m < all_tokens_list[i].Count; m++)
                        {
                            if (m >= prompt_reset_since_list[i] || m <= 4)
                            {
                                _decodingOptions.prompt.Add(all_tokens_list[i][m]);
                            }
                        }
                    }
                    else
                    {
                        _decodingOptions.prompt = decodingPromptList[i];
                    }
                    string language = _decodingOptions.language;
                    if (detectLanguageEntity.LanguageCodes.Count == batchSize)
                    {
                        language = detectLanguageEntity.LanguageCodes[i];
                    }
                    currLanguages.Add(language);
                    List<Int64> initialTokens = GetInitTokens(language: language).ToList();
                    initialTokensList.Add(initialTokens);
                }
                int framesPerSecond = WhisperFeatures.FramesPerSecond;
                // seek_clips[clip_idx].Item1;
                // mel frames per output token: 2
                int input_stride = ComputeHelper.ExactDiv(WhisperFeatures.NFrames, _modelDimensions.n_audio_ctx);
                // time per output token: 0.02 (seconds)
                float time_precision = (float)input_stride * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;
                //int prompt_reset_since = 0;
                //float last_speech_timestamp = 0.0f;
                int segment_size = WhisperFeatures.NFrames;
                float segment_duration = segment_size * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;

                // 计算 mel->audio_features
                DecodingResultEntity decodingResultEntity = DecodeWithFallback(encoderOutputEntity, initialTokensList, batchSize);

                int seek = seekList[0];
                int shiftLength = 0;// 计算本次解码的结束位置
                int removeOffset = 0;// seek > 3000 && seek % 3000 == 0 ? seek / 3000 * 2300 : 0;// 2300 是动态的，设置为上次输入的时长
                float time_offset = (seek - removeOffset) * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;
                float window_end_time = (seek * WhisperFeatures.NFrames) * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;
                bool should_skip = false;
                if (_no_speech_threshold != null)
                {
                    // no voice activity check
                    should_skip = decodingResultEntity.NoSpeechProb > _no_speech_threshold;
                    if (_logprob_threshold != float.NaN && decodingResultEntity.AvgLogprob > _logprob_threshold)
                    {
                        should_skip = false;
                    }
                }
                if (should_skip)
                {
                    seek += segment_size;
                    shiftLength = segment_size;
                }
                else
                {
                    List<SegmentEntity> current_segments = new List<SegmentEntity>();
                    // anomalous words are very long/short/improbable
                    List<bool> timestamp_tokens = decodingResultEntity.Tokens.Select(x => x >= _tokenizer.TimestampBegin).ToList();
                    bool single_timestamp_ending = false;
                    List<int> consecutive = new List<int>();
                    if (timestamp_tokens.Count >= 2)
                    {
                        single_timestamp_ending = timestamp_tokens.Last() == true && timestamp_tokens[timestamp_tokens.Count - 2] == false;

                        List<bool> timestamp_tokens_1 = new List<bool>(timestamp_tokens);
                        timestamp_tokens_1.Remove(timestamp_tokens_1.Last());
                        List<bool> timestamp_tokens_2 = new List<bool>(timestamp_tokens);
                        timestamp_tokens_2.Remove(timestamp_tokens_2.First());
#if NET6_0_OR_GREATER
                        int i = 0;
                        foreach (var item in timestamp_tokens_1.Zip<bool, bool>(timestamp_tokens_2))
                        {
                            if (item.First & item.Second)
                            {
                                consecutive.Add(i + 1);
                            }
                            i++;
                        }
#else
                        for (int n = 0; n < timestamp_tokens_1.Count && n < timestamp_tokens_2.Count; n++)
                        {
                            bool t1 = timestamp_tokens_1[n];
                            bool t2 = timestamp_tokens_2[n];
                            if (t1 & t2)
                            {
                                consecutive.Add(n + 1);
                            }
                            n++;
                        }
#endif
                    }

                    if (consecutive.Count > 0)
                    {
                        // if the output contains two consecutive timestamp tokens
                        if (single_timestamp_ending)
                        {
                            //consecutive.Add(decodingResultEntity.Tokens.Count - 1);
                            consecutive.Add(decodingResultEntity.Tokens.Count);
                        }
                        int lastSlice = 0;
                        foreach (var currentSlice in consecutive)
                        {
                            List<int> sliced_tokens = new List<int>();
                            for (int i = 0; i < decodingResultEntity.Tokens.Count; i++)
                            {
                                if (i >= lastSlice && i <= currentSlice)
                                {
                                    int token = decodingResultEntity.Tokens[i];
                                    sliced_tokens.Add(token);
                                }
                            }
                            int start_timestamp_pos = sliced_tokens[0] - _tokenizer.TimestampBegin;
                            int end_timestamp_pos = sliced_tokens.Last() - _tokenizer.TimestampBegin;
                            float start = time_offset + start_timestamp_pos * time_precision;
                            float end = time_offset + end_timestamp_pos * time_precision;
                            current_segments.Add(NewSegment(seek, start, end, sliced_tokens, currLanguages[0], decodingResultEntity));
                            lastSlice = currentSlice;
                        }
                        if (single_timestamp_ending)
                        {
                            // single timestamp at the end means no speech after the last timestamp.
                            seek += segment_size;
                            shiftLength = segment_size;
                        }
                        else
                        {
                            // otherwise, ignore the unfinished segment and seek to the last timestamp
                            int last_timestamp_pos = decodingResultEntity.Tokens[lastSlice - 1] - _tokenizer.TimestampBegin;
                            seek += last_timestamp_pos * input_stride;
                            shiftLength = last_timestamp_pos * input_stride;
                        }
                        //if (single_timestamp_ending)
                        //{
                        //    // single timestamp at the end means no speech after the last timestamp.
                        //    //seek += segment_size;
                        //    int last_timestamp_pos = decodingResultEntity.Tokens[lastSlice - 1] - _tokenizer.TimestampBegin;
                        //    seek += last_timestamp_pos * input_stride;
                        //}
                        //else
                        //{
                        //    // otherwise, ignore the unfinished segment and seek to the last timestamp
                        //    int last_timestamp_pos = decodingResultEntity.Tokens[lastSlice - 1] - _tokenizer.TimestampBegin;
                        //    seek += last_timestamp_pos * input_stride;
                        //}
                    }
                    else
                    {
                        float duration = segment_duration;
                        List<int> timestamps = new List<int>();
                        //timestamp_tokens.Clear();
                        for (int i = 0; i < timestamp_tokens.Count; i++)
                        {
                            if (timestamp_tokens[i])
                            {
                                timestamps.Add(decodingResultEntity.Tokens[i]);
                            }
                            //timestamps.Add(decodingResultEntity.Tokens.Last());
                        }
                        if (timestamps.Count > 0 && timestamps.Last() != _tokenizer.TimestampBegin)
                        {
                            // no consecutive timestamps but it has a timestamp; use the last one.
                            int last_timestamp_pos = timestamps.Last() - _tokenizer.TimestampBegin;
                            duration = last_timestamp_pos * time_precision;
                        }
                        //else
                        //{
                        //    seek += segment_size;
                        //}
                        float start = time_offset;
                        float end = time_offset + duration;
                        current_segments.Add(NewSegment(seek, start, end, decodingResultEntity.Tokens, currLanguages[0], decodingResultEntity));
                        //seek += (int)(duration/ time_precision);
                        seek += segment_size;
                        shiftLength = segment_size;
                    }
                    if (_word_timestamps)
                    {
                        // TODO:L385

                    }
                    // if a segment is instantaneous or does not contain text, clear it
                    for (int i = 0; i < current_segments.Count; i++)
                    {
                        SegmentEntity segment = current_segments[i];
                        if (segment.Start == segment.End || string.IsNullOrEmpty(segment.Text))
                        {
                            //segment.Text = "";
                            //segment.Tokens = new List<int>();
                            //segment.Words = new List<Dictionary<string, string>>();
                            current_segments.Remove(segment);
                        }
                    }
                    all_segments_list[0] = current_segments;
                    foreach (var segment in current_segments)
                    {
                        all_tokens_list[0].AddRange(segment.Tokens);
                    }
                    if (!_condition_on_previous_text || decodingResultEntity.Temperature > 0.5)
                    {
                        //prompt_reset_since_list[0] = all_tokens_list[0].Count;
                        decodingPromptList[0] = new List<int>(initial_prompt_tokens);
                        prompt_reset_since_list[0] = 0;
                        _decodingOptions.temperature = 0.0f;
                    }
                    else
                    {
                        if (currLanguages[0] == previousLanguages[0])
                        {
                            //_decodingOptions.prompt = new List<int>(all_tokens_list[0]);
                            decodingPromptList[0] = new List<int>();
                            for (int i = 0; i < all_tokens_list[0].Count; i++)
                            {
                                if (i >= prompt_reset_since_list[0])
                                {
                                    decodingPromptList[0].Add(all_tokens_list[0][i]);
                                }
                            }
                            prompt_reset_since_list[0] = all_tokens_list[0].Count;
                        }
                        else
                        {
                            //_decodingOptions.prompt = new List<int>(initial_prompt_tokens);
                            decodingPromptList[0] = new List<int>(initial_prompt_tokens);
                            prompt_reset_since_list[0] = 0;
                            _decodingOptions.temperature = 0.0f;
                        }
                    }
                }
                seekList[0] = seek;

                // into stream
                int streamIndex = 0;
                foreach (OnlineStream stream in streamsWorking)
                {
                    stream.Tokens = all_tokens_list[streamIndex];
                    stream.AllSegments = all_segments_list[streamIndex];
                    stream.Seek = seekList[streamIndex];
                    stream.Prompt_reset_since = prompt_reset_since_list[streamIndex];
                    stream.DecodingPrompt = decodingPromptList[streamIndex];
                    stream.RemoveDecodedChunk(shiftLength);
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }
        }

        private SegmentEntity NewSegment(int seek, float start, float end, List<int> tokens, string language, DecodingResultEntity result)
        {
            tokens = tokens.Where(x => x < _tokenizer.Eot).ToList();
            SegmentEntity segmentEntity = new SegmentEntity();
            segmentEntity.Seek = seek;
            segmentEntity.Start = start;
            segmentEntity.End = end;
            segmentEntity.Tokens = tokens;
            segmentEntity.Text = _tokenizer.decode(tokens);
            segmentEntity.Temperature = result.Temperature;
            segmentEntity.AvgLogprob = result.AvgLogprob;
            segmentEntity.CompressionRatio = result.CompressionRatio;
            segmentEntity.NoSpeechProb = result.NoSpeechProb;
            segmentEntity.Language = language;
            return segmentEntity;
        }

        private float WordAnomalyScore(Dictionary<string, string> word)
        {
            float probability = 0.0f;
            //word.TryGetValue("probability", out probability);
            float.TryParse(word.GetValueOrDefault("probability"), out probability);
            float end = 0.0f;
            float start = 0.0f;
            float.TryParse(word.GetValueOrDefault("end"), out end);
            float.TryParse(word.GetValueOrDefault("start"), out start);
            float duration = end - start;
            float score = 0.0f;
            if (probability < 0.15f)
            {
                score += 1.0f;
            }
            if (probability < 0.133f)
            {
                score += (0.133f - duration) * 15;
            }
            if (probability > 2.0f)
            {
                score += duration - 2.0f;
            }
            return score;
        }

        private bool IsSegmentAnomaly(SegmentEntity segmentEntity)
        {
            if (segmentEntity == null || segmentEntity.Words == null) return false;
            Dictionary<string, string>[] words = segmentEntity.Words.Where(x => !_punctuation.ToCharArray().Select(x => x.ToString()).Contains(x.GetValueOrDefault("word"))).ToArray();
            Dictionary<string, string>[] newWords = new Dictionary<string, string>[8];
            Array.Copy(words, 0, newWords, 0, newWords.Length);
            words = newWords;
            List<float> scores = new List<float>();
            foreach (var word in words)
            {
                float score = WordAnomalyScore(word);
                scores.Add(score);
            }
            float scoreSum = scores.Sum();
            return scoreSum >= 3 || scoreSum + 0.01f >= words.Length;
        }

        private Dictionary<string, string> NextWordsSegment(List<Dictionary<string, string>> segments)
        {
            Dictionary<string, string> s = new Dictionary<string, string>();
            // TODO: get next
            return s;
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
                if (tokens[i].Last() == eot || sumLogprobs[i] == float.NegativeInfinity)
                {
                    nextTokens[i][nextTokens[i].Count - 1] = eot;
                }
            }
            //tokens = torch.cat([tokens, next_tokens[:, None]], dim = -1)
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
            //ConfEntity? confEntity = _onlineProj?.ConfEntity;
            if (_modelDimensions != null)
            {
                //
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
                //
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
        /// <summary>
        /// MaximumLikelihoodRanker
        /// sequence ranker: implements how to rank a group of sampled sequences
        /// </summary>
        private List<List<float>> MaximumLikelihoodRanker(float lengthPenalty, List<List<List<int>>> tokensList, List<List<float>> sumLogprobsList)
        {
            List<List<float>> results = new List<List<float>>();
#if NET6_0_OR_GREATER
            foreach (var items in tokensList.Zip<List<List<int>>, List<float>>(sumLogprobsList))
            {
                List<List<int>> tokens = items.First;
                List<float> sumLogprobs = items.Second;
#else
            for (int i = 0; i < tokensList.Count && i < sumLogprobsList.Count; i++)
            {
                List<List<int>> tokens = tokensList[i];
                List<float> sumLogprobs = sumLogprobsList[i];
#endif
                int[] lengths = tokens.Select(x => x.Count).ToArray();
                List<float> result = new List<float>();
#if NET6_0_OR_GREATER
                foreach (var item in sumLogprobs.Zip<float, int>(lengths))
                {
                    float logprob = item.First;
                    int length = item.Second;
#else
                for (int j = 0; j < sumLogprobs.Count && j < lengths.Length; j++)
                {
                    float logprob = sumLogprobs[j];
                    int length = lengths[j];
#endif
                    float penalty = float.NaN;
                    if (lengthPenalty == null)
                    {
                        penalty = length;
                    }
                    else
                    {
                        // from the Google NMT paper
                        penalty = (float)Math.Pow(((5 + length) / 6), (double)lengthPenalty);
                    }
                    result.Add(penalty);
                }
                results.Add(result);
            }
            return results;
        }

        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OnlineStream stream in streams)
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
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                //english;
                text_result = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", " ").Replace("▁", "").ToLower();
                //text_result = englishSpellingNormalizer.GetNormalizerText(text_result);
                onlineRecognizerResultEntity.Text = text_result;
                onlineRecognizerResultEntity.Tokens = stream.Tokens;
                onlineRecognizerResultEntity.Segments = stream.AllSegments;
                //onlineRecognizerResultEntity.Timestamps = stream.Timestamps;
                //onlineRecognizerResultEntity.Language = stream.Language;
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return onlineRecognizerResultEntities;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onlineProj != null)
                    {
                        _onlineProj.Dispose();
                    }
                    if (_onlineModel != null)
                    {
                        _onlineModel.Dispose();
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
        ~OnlineRecognizer()
        {
            Dispose(_disposed);
        }
    }
}