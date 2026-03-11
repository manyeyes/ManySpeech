// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.FireRedAsr.Model;
using System.Text.RegularExpressions;

namespace ManySpeech.FireRedAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    public class LanguageDetection : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private string _mvnFilePath;
        private IOfflineProj _offlineProj;

        public LanguageDetection(string encoderFilePath, string decoderFilePath, string mvnFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 1)
        {
            _mvnFilePath = mvnFilePath;
            OfflineModel offlineModel = new OfflineModel(encoderFilePath, decoderFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _offlineProj = new LidProjOfAED(offlineModel);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_mvnFilePath, _offlineProj);
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
            this.Forward(streams);
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            int contextSize = 1;
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<List<Int64>> tokensList = new List<List<Int64>>();
            List<List<int[]>> timestampsList = new List<List<int[]>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                OfflineInputEntity asrInputEntity = new OfflineInputEntity();

                asrInputEntity.Speech = stream.GetDecodeChunk();
                if (asrInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                asrInputEntity.SpeechLength = asrInputEntity.Speech.Length;
                modelInputs.Add(asrInputEntity);
                statesList.Add(stream.States);
                tokensList.Add(stream.Tokens);
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
                stackStatesList = _offlineProj.stack_states(statesList);
                EncoderOutputEntity encoderOutputEntity = _offlineProj.EncoderProj(modelInputs);
                // conf args
                int beamSize = 1;
                int nbest = 1;
                int decode_max_len = 0;
                float softmax_smoothing = 1.25F;
                float length_penalty = 0.6F;
                float eos_penalty = 1.0F;
                // Init
                int N = batchSize;
                int H = 1280;
                int Ti = encoderOutputEntity.Output.Length / N / H;
                int maxlen = decode_max_len > 0 ? decode_max_len : Ti;
                // encoder
                float[] encoder_outputs = encoderOutputEntity.Output;
                bool[] src_mask = encoderOutputEntity.Mask;
                // decoder
                if (_offlineProj.DecoderSession != null)
                {
                    for (int i = 0; i < maxlen; i++)
                    {
                        DecoderOutputEntity decoderOutputEntity = _offlineProj.DecoderProj(tokensList, encoder_outputs, src_mask, stackStatesList);
                        // 合并对应索引的子列表
                        tokensList = tokensList.Zip(decoderOutputEntity.TokensList, (a, b) => a.Concat(b).ToList()).ToList();
                        stackStatesList = decoderOutputEntity.CacheList;
                        bool allEnd = tokensList.All(item => item.Any() && item.Last() == _offlineProj.Eos_id);
                        if (allEnd)
                        {
                            break;
                        }
                    }
                }
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.Tokens = tokensList[streamIndex].ToList();
                    stream.RemoveDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }

        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
            List<string> text_results = new List<string>();
#pragma warning disable CS8602 // 解引用可能出现空引用。

            foreach (var stream in streams)
            {
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                string lastToken = "";
#if NET6_0_OR_GREATER
                foreach (var result in stream.Tokens)
                {
                    Int64 token = result;
#else
                for (int i = 0; i < stream.Tokens.Count; i++)
                {
                    Int64 token = stream.Tokens[i];
#endif
                    if (token == 2)
                    {
                        break;
                    }
                    string currText = _tokens[token].Split(new char[] { '\t', ' ' })[0].ToLower();
                    if (currText != "</s>" && currText != "<s>" && currText != "<sos/eos>" && currText != "<blank>" && currText != "<unk>" && currText != "<sos>" && currText != "<eos>" && currText != "<pad>")
                    {
                        offlineRecognizerResultEntity.Tokens.Add(currText);
                    }
                }
                offlineRecognizerResultEntity.Language = offlineRecognizerResultEntity.Tokens?.First() ?? "";
                offlineRecognizerResultEntities.Add(offlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。
            return offlineRecognizerResultEntities;
        }

        public void DisposeOfflineStream(OfflineStream offlineStream)
        {
            if (offlineStream != null)
            {
                offlineStream.Dispose();
            }
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
        ~LanguageDetection()
        {
            Dispose(_disposed);
        }
    }
}