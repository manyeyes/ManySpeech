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
    public class LanguageDetection:IDisposable
    {
        private bool _disposed;
        private Tokenizer _tokenizer;
        private IOfflineProj? _offlineProj;
        private OfflineModel _offlineModel;
        private bool _isDetectLanguage = false;
        private DecodingOptions? _decodingOptions;
        private ModelDimensions? _modelDimensions;
        public OfflineModel OfflineModel { get => _offlineModel; set => _offlineModel = value; }

        public LanguageDetection(string encoderFilePath, string decoderFilePath, ConfEntity? confEntity=null, string configFilePath = "", int threadsNum = 1)
        {
            try
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
                OfflineModel = new OfflineModel(encoderFilePath, decoderFilePath, confEntity: confEntity, threadsNum: threadsNum);
                _isDetectLanguage = confEntity?.is_multilingual ?? false;
                if (!_isDetectLanguage)
                {
                    Console.WriteLine(string.Format("Current model is an English-only model. Unable to perform language detection"));
                    throw new Exception(string.Format("Current model is an English-only model. Unable to perform language detection"));
                }
                _decodingOptions = confEntity?.decoding_options;
                _modelDimensions = confEntity?.model_dimensions;
                if (_decodingOptions == null)
                {
                    _decodingOptions = new DecodingOptions();
                }
                _tokenizer = GetTokenizer(multilingual: confEntity.is_multilingual, language: _decodingOptions.language, task: _decodingOptions.task, numLanguages: confEntity.num_languages);
                _offlineProj = new OfflineProj(OfflineModel);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString);
            }
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
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                OfflineInputEntity offlineInputEntity = new OfflineInputEntity();

                offlineInputEntity.Speech = stream.GetDecodeChunk();
                offlineInputEntity.SampleLength = stream.RealSampleLen;
                if (offlineInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                offlineInputEntity.SpeechLength = offlineInputEntity.Speech.Length;
                modelInputs.Add(offlineInputEntity);
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
                EncoderOutputEntity encoderOutputEntity = _offlineProj.EncoderProj(modelInputs);
                // 检测语种
                DetectLanguageEntity detectLanguageEntity = new DetectLanguageEntity();
                if (_isDetectLanguage)
                {
                    detectLanguageEntity = DetectLanguage(encoderOutputEntity);
                }
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.Language = detectLanguageEntity.LanguageCodes[streamIndex];
                    stream.RemoveAllDecodedChunk();
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
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                offlineRecognizerResultEntity.Language = stream.Language;
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
        ~LanguageDetection()
        {
            Dispose(_disposed);
        }
    }
}