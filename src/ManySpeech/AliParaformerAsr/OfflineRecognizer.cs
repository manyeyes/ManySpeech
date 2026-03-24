// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using System.Text.RegularExpressions;

namespace ManySpeech.AliParaformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private IOfflineProj _offlineProj;

        public OfflineRecognizer(string modelFilePath, string configFilePath, string mvnFilePath, string tokensFilePath, string embedFilePath = "", string hotwordFilePath = "", int threadsNum = 1)
        {
            OfflineModel offlineModel = new OfflineModel(modelFilePath: modelFilePath, tokensFilePath: tokensFilePath, mvnFilePath: mvnFilePath, configFilePath: configFilePath, hotwordFilePath: hotwordFilePath, embedFilePath: embedFilePath, threadsNum: threadsNum);
            switch (offlineModel.ConfEntity.model.ToLower())
            {
                case "paraformer":
                    _offlineProj = new OfflineProjOfParaformer(offlineModel);
                    break;
                case "sensevoicesmall":
                    _offlineProj = new OfflineProjOfSenseVoiceSmall(offlineModel);
                    break;
                case "seacoparaformer":
                    _offlineProj = new OfflineProjOfSeacoParaformer(offlineModel);
                    break;
                default:
                    _offlineProj = new OfflineProjOfParaformer(offlineModel);
                    break;
            }
        }

        public OfflineStream CreateOfflineStream()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException("OfflineRecognizer");
            }
            OfflineStream offlineStream = new OfflineStream(_offlineProj);
            return offlineStream;
        }

        public OfflineRecognizerResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            OfflineRecognizerResultEntity result = GetResults(streams)[0];

            return result;
        }
        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            ProcessStreams(streams);
            List<OfflineRecognizerResultEntity> results = this.DecodeMulti(streams);
            return results;
        }

        private void ProcessStreams(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();  
            List<List<int>> tokenIdsList = new List<List<int>>();
            List<List<int[]>> timestampsList = new List<List<int[]>>();
            foreach (OfflineStream stream in streams)
            {
                var decodeChunk = stream.GetDecodeChunk();
                modelInputs.Add(decodeChunk);
            }
            try
            {
                _offlineProj.Infer(modelInputs, tokenIdsList, timestampsList);
                int streamIndex = 0;
                foreach (OfflineStream stream in streams)
                {
                    stream.TokenIds = tokenIdsList[streamIndex].ToList();
                    stream.Timestamps = timestampsList[streamIndex];
                    stream.RemoveChunk();
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
                for (int i = 0; i < stream.Tokens.Count && i < stream.Timestamps.Count; i++)
                {
                    int tokenId = stream.TokenIds[i];
                    int[] timestamp = stream.Timestamps[i];
#endif
                    if (tokenId == 2)
                    {
                        break;
                    }
                    string currText = _offlineProj.OfflineModel.Tokens[tokenId].Split('\t')[0];
                    if (currText != "</s>" && currText != "<s>" && currText != "<blank>" && currText != "<unk>")
                    {
                        if (IsChinese(currText, true))
                        {
                            text_result += currText;
                            offlineRecognizerResultEntity.Tokens.Add(currText);
                            offlineRecognizerResultEntity.Timestamps.Add(timestamp);
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
                                offlineRecognizerResultEntity.Tokens.Remove(offlineRecognizerResultEntity.Tokens.Last());
                                offlineRecognizerResultEntity.Tokens.Add(currToken.Replace("▁", ""));
                                offlineRecognizerResultEntity.Timestamps.Remove(offlineRecognizerResultEntity.Timestamps.Last());
                                offlineRecognizerResultEntity.Timestamps.Add(currTimestamp);
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
                                if (offlineRecognizerResultEntity.Tokens.Count > 0)
                                {
                                    offlineRecognizerResultEntity.Tokens.Remove(offlineRecognizerResultEntity.Tokens.Last());
                                }
                                offlineRecognizerResultEntity.Tokens.Add(currToken.Replace("▁", ""));
                                if (offlineRecognizerResultEntity.Timestamps.Count > 0)
                                {
                                    offlineRecognizerResultEntity.Timestamps.Remove(offlineRecognizerResultEntity.Timestamps.Last());
                                }
                                offlineRecognizerResultEntity.Timestamps.Add(currTimestamp);
                                lastToken = currToken;
                                lastTimestamp = currTimestamp;
                            }
                            else
                            {
                                offlineRecognizerResultEntity.Tokens.Add(currText.Replace("▁", ""));
                                offlineRecognizerResultEntity.Timestamps.Add(timestamp);
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
                //offlineRecognizerResultEntity.TokenIds = stream.TokenIds;
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
                        _offlineProj = null;
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