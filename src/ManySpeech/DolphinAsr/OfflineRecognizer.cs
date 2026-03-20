// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.DolphinAsr.Model;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace ManySpeech.DolphinAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private IOfflineProj _offlineProj;

        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 2)
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
                // If specified language and region.
                if (stream.TokenIds.Count == 1 && !string.IsNullOrEmpty(stream.Language) && !string.IsNullOrEmpty(stream.Region))
                {
                    int langId = Array.IndexOf(_tokens, $"<{stream.Language.ToLower()}>");
                    int regionId = Array.IndexOf(_tokens, $"<{stream.Region.ToUpper()}>");
                    if (langId > 0 && regionId > 0)
                    {
                        stream.TokenIds.Add(langId);
                        stream.TokenIds.Add(regionId);
                        stream.TokenIds.Add(_offlineProj.OfflineModel.AsrId);
                    }
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
                    for(int i = 0;i< detectLangIdsList.Count; i++)
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
                // conf args
                int beamSize = 1;
                int nbest = 1;
                int decodeMaxLen = 0;
                float softmaxSmoothing = 1.25F;
                float lengthPenalty = 0.6F;
                float eosPenalty = 1.0F;
                // Init
                int N = batchSize;
                int H = 512;
                int Ti = encoderOutputEntity.Output.Length / N / H;
                int maxlen = decodeMaxLen > 0 ? decodeMaxLen : Ti;
                
                // decoder
                if (_offlineProj.DecoderSession != null)
                {
                    for (int i = 0; i < maxlen; i++)
                    {
                        DecoderOutputEntity decoderOutputEntity = _offlineProj.DecoderProj(tokenIdsList, encoder_outputs);
                        // 合并对应索引的子列表
                        tokenIdsList = tokenIdsList.Zip(_offlineProj.DecodeAsr(decoderOutputEntity.LogitsTensor), (a, b) => a.Concat(b).ToList()).ToList();
                        bool allEnd = tokenIdsList?.All(list => list.Contains(_offlineProj.OfflineModel.EosId)) ?? false;
                        if (allEnd)
                        {
                            break;
                        }
                    }
                }
                for (int i = 0; i < batchSize; i++)
                {
                    List<int> tokens = tokenIdsList[i].Select(x => (int)x).ToList();
                    // 找到第一个 Eos_id 的索引
                    int eosIndex = tokens.FindIndex(t => t == _offlineProj.OfflineModel.EosId);
                    // 如果找到 Eos_id，移除它及其后面的所有元素
                    if (eosIndex != -1)
                    {
                        tokens.RemoveRange(eosIndex, tokens.Count - eosIndex);
                    }
                    List<int[]> timestamps = new List<int[]>();
                    for (int j = 0; j < tokens.Count; j++)
                    {
                        timestamps.Add(new int[2]);
                    }
                    timestampsList[i].AddRange(timestamps);
                }
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.TokenIds = tokenIdsList[streamIndex].ToList();
                    stream.Timestamps = timestampsList[streamIndex];
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
                string text_result = "";
                string lastToken = "";
                int[] lastTimestamp = null;
#if NET6_0_OR_GREATER
                foreach (var result in stream.TokenIds.Zip<int, int[]>(stream.Timestamps))
                {
                    int tokenId = result.First;
                    int[] timestamp = result.Second;
#else
                for (int i = 0; i < stream.TokenIds.Count && i < stream.Timestamps.Count; i++)
                {
                    int tokenId = stream.TokenIds[i];
                    int[] timestamp = stream.Timestamps[i];
#endif
                    if (tokenId == 2)
                    {
                        break;
                    }
                    string currText = _tokens[tokenId].Split(new char[] { '\t', ' ' })[0];
                    offlineRecognizerResultEntity.Tokens.Add(currText);
                    if (currText != "</s>" && currText != "<s>" && currText != "<sos/eos>" && currText != "<blank>" && currText != "<unk>" && currText != "<sos>" && currText != "<eos>" && currText != "<pad>")
                    {
                        if (IsChinese(currText, true))
                        {
                            text_result += currText;
                            offlineRecognizerResultEntity.Words.Add(currText);
                            offlineRecognizerResultEntity.WordsTimestamps.Add(timestamp);
                        }
                        else
                        {
                            text_result += "▁" + currText + "▁";
                            if ((lastToken + "▁" + currText + "▁").IndexOf("@@▁▁") > 0)
                            {
                                string currToken = (lastToken + "▁" + currText + "▁").Replace("@@▁▁", "");
                                int[] currTimestamp = null;
                                if (lastTimestamp == null)
                                {
                                    currTimestamp = timestamp;
                                }
                                else
                                {
                                    List<int> temp = lastTimestamp.ToList();
                                    temp.AddRange(timestamp.ToList());
                                    currTimestamp = temp.ToArray();
                                }
                                offlineRecognizerResultEntity.Words.Remove(offlineRecognizerResultEntity.Words.Last());
                                offlineRecognizerResultEntity.Words.Add(currToken.Replace("▁", ""));
                                offlineRecognizerResultEntity.WordsTimestamps.Remove(offlineRecognizerResultEntity.WordsTimestamps.Last());
                                offlineRecognizerResultEntity.WordsTimestamps.Add(currTimestamp);
                                lastToken = currToken;
                                lastTimestamp = currTimestamp;
                            }
                            else if (((lastToken + "▁" + currText + "▁").Count(x => x == '▁') == 3 || (lastToken + "▁" + currText + "▁").Count(x => x == '▁') == 5) && (lastToken + "▁" + currText + "▁").IndexOf("▁▁▁") < 0)
                            {
                                string currToken = (lastToken + "▁" + currText + "▁").Replace("▁▁", "");
                                int[] currTimestamp = null;
                                if (lastTimestamp == null)
                                {
                                    currTimestamp = timestamp;
                                }
                                else
                                {
                                    List<int> temp = lastTimestamp.ToList();
                                    temp.AddRange(timestamp.ToList());
                                    currTimestamp = temp.ToArray();
                                }
                                if (offlineRecognizerResultEntity.Words.Count > 0)
                                {
                                    offlineRecognizerResultEntity.Words.Remove(offlineRecognizerResultEntity.Words.Last());
                                }
                                offlineRecognizerResultEntity.Words.Add(currToken.Replace("▁", ""));
                                if (offlineRecognizerResultEntity.WordsTimestamps.Count > 0)
                                {
                                    offlineRecognizerResultEntity.WordsTimestamps.Remove(offlineRecognizerResultEntity.WordsTimestamps.Last());
                                }
                                offlineRecognizerResultEntity.WordsTimestamps.Add(currTimestamp);
                                lastToken = currToken;
                                lastTimestamp = currTimestamp;
                            }
                            else
                            {
                                offlineRecognizerResultEntity.Words.Add(currText.Replace("▁", ""));
                                offlineRecognizerResultEntity.WordsTimestamps.Add(timestamp);
                                lastToken = "▁" + currText + "▁";
                                lastTimestamp = timestamp;
                            }

                        }

                    }
                }
                if (text_result.IndexOf("@@▁▁") > 0 || text_result.IndexOf("▁▁▁") < 0)
                {
                    text_result = text_result.Replace("@@▁▁", "").Replace("▁▁", " ").Replace("@@", " ").Replace("▁", " ");
                }
                else
                {
                    text_result = text_result.Replace("▁▁▁", " ").Replace("▁▁", "").Replace("▁", "");
                }
                text_results.Add(text_result);
                offlineRecognizerResultEntity.Region = stream.Region;
                offlineRecognizerResultEntity.Language = stream.Language;
                offlineRecognizerResultEntity.Timestamps = stream.Timestamps;
                offlineRecognizerResultEntity.Text = text_result;
                offlineRecognizerResultEntities.Add(offlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。
            return offlineRecognizerResultEntities;
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
        ~OfflineRecognizer()
        {
            Dispose(_disposed);
        }
    }
}