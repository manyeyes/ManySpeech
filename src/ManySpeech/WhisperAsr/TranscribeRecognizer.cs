// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;
using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr.Utils;

namespace ManySpeech.WhisperAsr
{
    /// <summary>
    /// transcribe recognizer package
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class TranscribeRecognizer : IDisposable
    {
        private bool _disposed;

        private Tokenizer _tokenizer;
        private ITranscribeProj? _transcribeProj;
        private TranscribeModel _transcribeModel;
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
        private bool _condition_on_previous_text = true;
        // 存储上一次的token,作为init token
        private string? _initial_prompt;
        private bool _word_timestamps = false;
        private string _prepend_punctuations = "\"'“¿([{-";
        private string _append_punctuations = "\"'.。,，!！?？:：”)]}、";
        private List<float> _clip_timestamps = new List<float>() { 0f };
        private float _hallucination_silence_threshold;
        private string _punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、";

        public TranscribeRecognizer(string encoderFilePath, string decoderFilePath, ConfEntity? confEntity = null, string configFilePath = "", int threadsNum = 1)
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
            _transcribeModel = new TranscribeModel(encoderFilePath, decoderFilePath, confEntity: confEntity ?? new ConfEntity(), threadsNum: threadsNum);

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
                _transcribeModel.SuppressSample = float.NaN; 
            }
            else
            {
                _isDetectLanguage = string.IsNullOrEmpty(_decodingOptions.language) ? true : false;
                _transcribeModel.SuppressSample = float.NegativeInfinity;
            }
            _decodingOptions.task = "transcribe";
            _tokenizer = GetTokenizer(multilingual: confEntity?.is_multilingual ?? false, language: _decodingOptions.language, task: _decodingOptions.task, numLanguages: confEntity?.num_languages ?? 99);
            _transcribeProj = new TranscribeProj(_transcribeModel);
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

        public TranscribeStream CreateTranscribeStream()
        {
            TranscribeStream transcribeStream = new TranscribeStream(_transcribeModel);
            return transcribeStream;
        }
        public TranscribeRecognizerResultEntity GetResult(TranscribeStream stream)
        {
            List<TranscribeStream> streams = new List<TranscribeStream>();
            streams.Add(stream);
            TranscribeRecognizerResultEntity transcribeRecognizerResultEntity = GetResults(streams)[0];

            return transcribeRecognizerResultEntity;
        }
        public List<TranscribeRecognizerResultEntity> GetResults(List<TranscribeStream> streams)
        {
            //this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<TranscribeRecognizerResultEntity> transcribeRecognizerResultEntities = this.DecodeMulti(streams);
            return transcribeRecognizerResultEntities;
        }

        private DetectLanguageEntity DetectLanguage(EncoderOutputEntity encoderOutputEntity)
        {
            DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
            DecoderOutputEntity decoderOutputEntityForDetectLanguage = new DecoderOutputEntity();
            decoderOutputEntityForDetectLanguage = _transcribeProj.DetectLanguage(encoderOutputEntity, _tokenizer.Sot);
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
                float precision = (float)_transcribeProj.ChunkLength / _modelDimensions.n_audio_ctx;
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
                DecoderOutputEntity decoderOutputEntity = _transcribeProj.DecoderProj(encoderOutputEntity, tokensParams);
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
                // if completed or tokens.shape[-1] > self.n_ctx:
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

        private void Forward(List<TranscribeStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<TranscribeStream> streamsWorking = new List<TranscribeStream>();
            List<List<SegmentEntity>> all_segments_list = new List<List<SegmentEntity>>();
            List<List<string>> all_languages_list=new List<List<string>>();
            List<TranscribeInputEntity> modelInputs = new List<TranscribeInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<List<int>> tokens = new List<List<int>>();
            List<TranscribeStream> streamsTemp = new List<TranscribeStream>();
            foreach (TranscribeStream stream in streams)
            {
                TranscribeInputEntity transcribeInputEntity = new TranscribeInputEntity();

                transcribeInputEntity.Speech = stream.GetAllDecodeChunk();
                transcribeInputEntity.SampleLength = stream.RealSampleLen;
                if (transcribeInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                transcribeInputEntity.SpeechLength = transcribeInputEntity.Speech.Length;
                modelInputs.Add(transcribeInputEntity);
                statesList.Add(stream.States);
                tokens.Add(stream.Tokens);
                streamsWorking.Add(stream);
                all_segments_list.Add(stream.AllSegments);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (TranscribeStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                int batchSize = modelInputs.Count;
                int featureDim = _modelDimensions.n_mels;
                int frameLength = WhisperFeatures.NFrames;
                int hopLength = WhisperFeatures.HopLength;
                int sampleRate = WhisperFeatures.SampleRate;
                int framesPerSecond = WhisperFeatures.FramesPerSecond;
                int content_frames = modelInputs[0].Speech.Length / _transcribeProj.FeatureDim <= frameLength ? frameLength : modelInputs[0].Speech.Length / _transcribeProj.FeatureDim;
                float content_duration = (float)(content_frames * hopLength / sampleRate);
                List<int> seek_points = _clip_timestamps.Select(x => (int)Math.Round(x * framesPerSecond)).ToList();
                if (seek_points.Count == 0)
                {
                    seek_points.Add(0);
                }
                if (seek_points.Count % 2 == 1)
                {
                    seek_points.Add(content_frames);
                }
                //seek_points[::2]
                List<int> seek_points_1 = new List<int>();
                for (int i = 0; i < seek_points.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        seek_points_1.Add(seek_points[i]);
                    }
                }
                //seek_points[1::2]
                List<int> seek_points_2 = new List<int>();
                for (int i = 0; i < seek_points.Count; i++)
                {
                    if (i % 2 == 1)
                    {
                        seek_points_2.Add(seek_points[i]);
                    }
                }
#if NET6_0_OR_GREATER
                List<(int, int)> seek_clips = seek_points_1.Zip<int, int>(seek_points_2).ToList();
#else
                // .NET 6.0以下版本
                // 初始化结果列表
                List<(int, int)> seek_clips = new List<(int, int)>();
                // 使用for循环遍历并配对元素
                for (int i = 0; i < seek_points_1.Count; i++)
                {
                    // 获取当前索引的元素
                    int item1 = seek_points_1[i];
                    int item2 = seek_points_2[i];

                    // 添加到结果列表（元组形式）
                    seek_clips.Add((item1, item2));
                }
#endif
                int clip_idx = 0;
                int seek = seek_clips[clip_idx].Item1;
                // mel frames per output token: 2
                int input_stride = ComputeHelper.ExactDiv(frameLength, _modelDimensions.n_audio_ctx);
                // time per output token: 0.02 (seconds)
                float time_precision = (float)input_stride * hopLength / sampleRate;
                List<List<int>> all_tokens_list = new List<List<int>>();
                int prompt_reset_since = 0;
                List<string> previousLanguages = new List<string>();
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
                for (int i = 0; i < batchSize; i++)
                {
                    if (all_tokens_list.Count == i)
                    {
                        all_tokens_list.Add(new List<int>());
                    }
                    all_tokens_list[i].AddRange(initial_prompt_tokens);
                    previousLanguages.Add("");
                    prompt_reset_since_list.Add(0);
                    decodingPromptList.Add(null);
                }
                while (clip_idx < seek_clips.Count)
                {
                    (int, int) seek_clip = seek_clips[clip_idx];
                    int seek_clip_start = seek_clip.Item1;
                    int seek_clip_end = seek_clip.Item2;
                    if (seek < seek_clip_start)
                    {
                        seek = seek_clip_start;
                    }
                    if (seek >= seek_clip_end)
                    {
                        clip_idx++;
                        if (clip_idx < seek_clips.Count)
                        {
                            seek = seek_clips[clip_idx].Item1;
                        }
                        continue;
                    }
                    int removeOffset = 0;// seek > 3000 && seek % 3000 == 0 ? seek / 3000 * 2300 : 0;// 2300 是动态的，设置为上次输入的时长
                    float time_offset = (seek - removeOffset) * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;
                    float window_end_time = (seek * WhisperFeatures.NFrames) * WhisperFeatures.HopLength / WhisperFeatures.SampleRate;
                    List<TranscribeInputEntity> transcribeInputEntities = new List<TranscribeInputEntity>();
                    int segment_size = new List<int>() { frameLength, content_frames - seek, seek_clip_end - seek }.Min();
                    float segment_duration = segment_size * hopLength / sampleRate;
                    for (int b = 0; b < batchSize; b++)
                    {
                        float[] mel_segment = new float[featureDim * frameLength];
                        TranscribeInputEntity transcribeInputEntity = new TranscribeInputEntity();
                        int oRowLen = modelInputs[b].Speech.Length / featureDim;
                        for (int n = 0; n < featureDim; n++)
                        {
                            Array.Copy(modelInputs[b].Speech, n * oRowLen + seek, mel_segment, n * frameLength, segment_size);
                        }
                        float[] firstRowChunk = new float[frameLength - 1];
                        Array.Copy(mel_segment, 0, firstRowChunk, 0, firstRowChunk.Length);
                        var firstRowAvg = firstRowChunk.Average();
                        int firstRowAvgNum = firstRowChunk.Where(x => x == firstRowAvg).ToArray().Length;
                        //dim min head length : 398
                        float[] headChunk = new float[398 * featureDim];
                        for (int i = 0; i < featureDim; i++)
                        {
                            Array.Copy(mel_segment, i * frameLength, headChunk, i * 398, 398);
                        }
                        var headAvg = headChunk.Average();
                        int headAvgNum = headChunk.Where(x => x == headAvg).ToArray().Length;
                        //dim min tail length : 398
                        float[] tailChunk = new float[398 * featureDim];
                        for (int i = 0; i < featureDim; i++)
                        {
                            Array.Copy(mel_segment, i * frameLength + frameLength - 400, tailChunk, i * 398, 398);
                        }
                        var tailAvg = tailChunk.Average();
                        int tailAvgNum = tailChunk.Where(x => x == tailAvg).ToArray().Length;
                        if (firstRowAvgNum == firstRowChunk.Length || headAvgNum == headChunk.Length)
                        {
                            mel_segment = mel_segment.Select(x => _transcribeModel.SuppressSample).ToArray();
                        }
                        else if (tailAvgNum == tailChunk.Length)
                        {
                            int len = 0;
                            for (int i = mel_segment.Length / featureDim - 1; i >= 0; i--)
                            {
                                if (mel_segment[i] == tailAvg)
                                {
                                    len++;
                                }
                                else
                                {
                                    break;
                                }
                            }
                            float[] tempChunk = new float[len];
                            //float epsilon = 1e-3f / 32768f;
                            //// 填充极小值（可选择交替符号避免直流偏移）
                            //for (int i = 0; i < tempChunk.Length; i++)
                            //{
                            //    // 每隔一个采样点取反，减少直流分量
                            //    tempChunk[i] = (i % 2 == 0) ? epsilon : -epsilon;
                            //}
                            tempChunk = tempChunk.Select(x => x == 0 ? -0.00070269317413142F : x).ToArray();
                            for (int i = 0; i < featureDim; i++)
                            {
                                Array.Copy(tempChunk, 0, mel_segment, i * frameLength + frameLength - len, Math.Min(len, 398 / 2));
                            }
                        }
                        transcribeInputEntity.Speech = mel_segment;
                        transcribeInputEntities.Add(transcribeInputEntity);
                    }
                    //_decodingOptions.prompt = new List<int>(all_tokens);
                    //for (int i = 0; i < _decodingOptions.prompt.Count; i++)
                    //{
                    //    if (i < prompt_reset_since)
                    //    {
                    //        _decodingOptions.prompt.Remove(_decodingOptions.prompt.First());
                    //    }
                    //}
                    // 计算 mel->audio_features
                    EncoderOutputEntity encoderOutputEntity_seek = _transcribeProj.EncoderProj(transcribeInputEntities);
                    // 检测语种
                    DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
                    if (_isDetectLanguage)
                    {
                        if (_isDetectLanguage)
                        {
                            detectLanguageEntity = DetectLanguage(encoderOutputEntity_seek);
                        }
                    }
                    // InitTokens                    
                    List<List<Int64>> initialTokensList = new List<List<long>>();
                    List<string> currLanguages = new List<string>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        //if (decodingPromptList[i] == null)
                        //{
                        //    _decodingOptions.prompt = new List<int>();
                        //    for (int m = 0; m < all_tokens_list[i].Count; m++)
                        //    {
                        //        if (m >= prompt_reset_since_list[i])
                        //        {
                        //            _decodingOptions.prompt.Add(all_tokens_list[i][m]);
                        //        }
                        //    }
                        //}
                        //else
                        //{
                        //    _decodingOptions.prompt = decodingPromptList[i];
                        //}
                        string language = _decodingOptions.language;
                        if (detectLanguageEntity.LanguageCodes.Count == batchSize)
                        {
                            language = detectLanguageEntity.LanguageCodes[i];
                        }
                        currLanguages.Add(language);
                        List<Int64> initialTokens = GetInitTokens(language: language).ToList();
                        initialTokensList.Add(initialTokens);
                    }
                    DecodingResultEntity decodingResultEntity = DecodeWithFallback(encoderOutputEntity_seek, initialTokensList, batchSize);

                    if (_no_speech_threshold != null)
                    {
                        // no voice activity check
                        bool should_skip = decodingResultEntity.NoSpeechProb > _no_speech_threshold;
                        if (_logprob_threshold != float.NaN && decodingResultEntity.AvgLogprob > _logprob_threshold)
                        {
                            should_skip = false;
                        }
                        if (should_skip)                   
                        {
                            seek += segment_size;
                            continue;
                        }
                    }
                    //int previous_seek = seek;
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
                                consecutive.Add(i + 1);//consecutive.Add(1);
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
                            current_segments.Add(NewSegment(seek, start, end, sliced_tokens, decodingResultEntity));
                            lastSlice = currentSlice;
                        }
                        if (single_timestamp_ending)
                        {
                            seek += segment_size;
                        }
                        else
                        {
                            int last_timestamp_pos = decodingResultEntity.Tokens[lastSlice - 1] - _tokenizer.TimestampBegin;
                            seek += last_timestamp_pos * input_stride;
                        }
                    }
                    else
                    {
                        float duration = segment_duration;
                        List<int> timestamps = new List<int>();
                        for (int i = 0; i < timestamp_tokens.Count; i++)
                        {
                            if (timestamp_tokens[i])
                            {
                                timestamps.Add(decodingResultEntity.Tokens[i]);
                            }
                        }
                        if (timestamps.Count > 0 && timestamps.Last() != _tokenizer.TimestampBegin)
                        {
                            // no consecutive timestamps but it has a timestamp; use the last one.
                            int last_timestamp_pos = timestamps.Last() - _tokenizer.TimestampBegin;
                            duration = last_timestamp_pos * time_precision;
                        }
                        float start = time_offset;
                        float end = time_offset + duration;
                        current_segments.Add(NewSegment(seek, start, end, decodingResultEntity.Tokens, decodingResultEntity));
                        seek += segment_size;
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
                    all_segments_list[0].AddRange(current_segments);
                    foreach (var segment in current_segments)
                    {
                        all_tokens_list[0].AddRange(segment.Tokens);
                    }
                    if (!_condition_on_previous_text || decodingResultEntity.Temperature > 0.5)
                    {
                        decodingPromptList[0] = new List<int>(initial_prompt_tokens);
                        prompt_reset_since_list[0] = 0;
                        _decodingOptions.temperature = 0.0f;
                    }
                    else
                    {
                        if (currLanguages[0] == previousLanguages[0])
                        {
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
                            decodingPromptList[0] = new List<int>(initial_prompt_tokens);
                            prompt_reset_since_list[0] = 0;
                            _decodingOptions.temperature = 0.0f;
                        }
                    }
                    previousLanguages = new List<string>(currLanguages);
                    //if (!_condition_on_previous_text || decodingResultEntity.Temperature > 0.5)
                    //{
                    //    prompt_reset_since = all_tokens_list[0].Count;
                    //}
                    //if (currLanguages[0] == previousLanguages[0])
                    //{
                    //    _decodingOptions.prompt = new List<int>(all_tokens_list[0]);
                    //    prompt_reset_since= all_tokens_list[0].Count;
                    //}
                    //else
                    //{
                    //    _decodingOptions.temperature = 0.0f;
                    //    prompt_reset_since = 0;
                    //    _decodingOptions.prompt = new List<int>(initial_prompt_tokens);
                    //}
                    ////_decodingOptions.prompt = new List<int>(all_tokens_list[0]);
                    //for (int i = 0; i < _decodingOptions.prompt.Count; i++)
                    //{
                    //    if (i < prompt_reset_since)
                    //    {
                    //        _decodingOptions.prompt.Remove(_decodingOptions.prompt.First());
                    //    }
                    //}
                }
                // into stream
                int streamIndex = 0;
                foreach (TranscribeStream stream in streamsWorking)
                {
                    stream.Tokens = all_tokens_list[streamIndex];
                    stream.AllSegments = all_segments_list[streamIndex];
                    stream.RemoveAllDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }
        }

        private SegmentEntity NewSegment(int seek, float start, float end, List<int> tokens, DecodingResultEntity result)
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
            //if (temperature == 0)
            //{
            //    Int64[] item = new Int64[logits_tensor_last.Count];
            //    for (int j = 0; j < logits_tensor_last.Count; j++)
            //    {
            //        int token = 0;
            //        for (int k = 1; k < logits_tensor_last[j].Length; k++)
            //        {
            //            token = logits_tensor_last[j][token] > logits_tensor_last[j][k] ? token : k;
            //        }
            //        item[j] = (int)token;
            //    }
            //    nextTokens.Add(item.ToList());
            //}
            //else
            //{
            //    //nextTokens = Categorical(logits=logits / self.temperature).sample()
            //}
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
                if (tokens[i].Last() == eot || sumLogprobs[i] == float.NegativeInfinity)//
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
            //ConfEntity? confEntity = _transcribeProj?.ConfEntity;
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
        // TODO:MaximumLikelihoodRanker
        private void MaximumLikelihoodRanker()
        {

        }

        private List<TranscribeRecognizerResultEntity> DecodeMulti(List<TranscribeStream> streams)
        {
            List<TranscribeRecognizerResultEntity> transcribeRecognizerResultEntities = new List<TranscribeRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (TranscribeStream stream in streams)
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
                TranscribeRecognizerResultEntity transcribeRecognizerResultEntity = new TranscribeRecognizerResultEntity();
                transcribeRecognizerResultEntity.Text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", " ").Replace("▁", "").ToLower();
                transcribeRecognizerResultEntity.Tokens = stream.Tokens;
                transcribeRecognizerResultEntity.Segments = stream.AllSegments;
                //transcribeRecognizerResultEntity.Timestamps = stream.Timestamps;
                //transcribeRecognizerResultEntity.Language = stream.Language;
                transcribeRecognizerResultEntities.Add(transcribeRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return transcribeRecognizerResultEntities;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_transcribeProj != null)
                    {
                        _transcribeProj.Dispose();
                    }
                    if (_transcribeModel != null)
                    {
                        _transcribeModel.Dispose();
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
        ~TranscribeRecognizer()
        {
            Dispose(_disposed);
        }
    }
}