// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.DolphinAsr.Model;

namespace ManySpeech.DolphinAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    public class LanguageID : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private IOfflineProj _offlineProj;

        public LanguageID(string encoderFilePath, string decoderFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 1)
        {
            OfflineModel offlineModel = new OfflineModel(encoderFilePath, decoderFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _offlineProj = new OfflineProjOfDolphin(offlineModel);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_offlineProj);
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
            List<List<int>> tokenIdsList = new List<List<int>>();
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
                if (stream.TokenIds.Count > 1)
                {
                    stream.TokenIds = stream.TokenIds.Take(1).ToList();
                }
                statesList.Add(stream.States);
                tokenIdsList.Add(stream.TokenIds);
                timestampsList.Add(stream.Timestamps);
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
                // encoder
                EncoderOutputEntity encoderOutputEntity = _offlineProj.EncoderProj(modelInputs);
                float[] encoder_outputs = encoderOutputEntity.Output;
                // If not specified, it will automatically detect lang and region.
                if (tokenIdsList.Min(x => x.Count) == 1)
                {
                    tokenIdsList = tokenIdsList.Select(innerList => innerList.Take(1).ToList()).ToList();
                    // detect language
                    DecoderOutputEntity decoderOutputEntity = _offlineProj.DecoderProj(tokenIdsList, encoder_outputs);
                    List<List<int>> detectLangIdsList = _offlineProj.DetectLanguage(decoderOutputEntity.LogitsTensor);
                    for (int i = 0; i < detectLangIdsList.Count; i++)
                    {
                        tokenIdsList[i].Add(detectLangIdsList[i][0]);
                    }
                    // detect region
                    decoderOutputEntity = _offlineProj.DecoderProj(tokenIdsList, encoder_outputs);
                    List<List<int>> detectRegionIdsList = _offlineProj.DetectRegion(decoderOutputEntity.LogitsTensor);
                    for (int i = 0; i < detectRegionIdsList.Count; i++)
                    {
                        tokenIdsList[i].Add(detectRegionIdsList[i][0]);
                    }
                    // Add elements (fixed format)
                    tokenIdsList = tokenIdsList.Select(inner => inner.Concat(new[] { _offlineProj.OfflineModel.AsrId }).ToList()).ToList();
                }
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.TokenIds = tokenIdsList[streamIndex].ToList();
                    stream.RemoveDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("LanguageID failed", ex);
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
                foreach (var result in stream.TokenIds)
                {
                    int tokenId = result;
#else
                for (int i = 0; i < stream.TokenIds.Count; i++)
                {
                    int tokenId = stream.TokenIds[i];
#endif
                    if (tokenId == 2)
                    {
                        break;
                    }
                    string currText = _tokens[tokenId].Split(new char[] { '\t', ' ' })[0];
                    if (currText != "</s>" && currText != "<s>" && currText != "<sos/eos>" && currText != "<blank>" && currText != "<unk>" && currText != "<sos>" && currText != "<eos>" && currText != "<pad>")
                    {
                        offlineRecognizerResultEntity.Tokens.Add(currText);
                    }
                }
                offlineRecognizerResultEntity.Language = offlineRecognizerResultEntity.Tokens?.Skip(0).First() ?? "";
                offlineRecognizerResultEntity.Region = offlineRecognizerResultEntity.Tokens?.Skip(1).First() ?? "";
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
        ~LanguageID()
        {
            Dispose(_disposed);
        }
    }
}