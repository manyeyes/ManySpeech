// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
using ManySpeech.OmniAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace ManySpeech.OmniAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private OfflineModel _offlineModel;
        private string[] _tokens;
        private ConfEntity _confEntity;
        private IOfflineProj _offlineProj;

        public OfflineRecognizer(string modelFilePath, string configFilePath, string tokensFilePath, int threadsNum = 1)
        {
            _offlineModel = new OfflineModel(modelFilePath: modelFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _confEntity = _offlineModel.ConfEntity;
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
            if (_tokens == null || _tokens.Length == 0)
            {
                throw new Exception("tokens invalid");
            }
            switch (_confEntity.model.ToLower())
            {
                case "omniasr-ctc":
                    _offlineProj = new OfflineProjOfOmniCtc(_offlineModel);
                    break;
                default:
                    _offlineProj = new OfflineProjOfOmniCtc(_offlineModel);
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
            OfflineRecognizerResultEntity text_result = GetResults(streams)[0];

            return text_result;
        }
        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            //this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OfflineRecognizerResultEntity> text_results = this.DecodeMulti(streams);
            return text_results;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            List<List<int>> tokensList = new List<List<int>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                OfflineInputEntity modelInputEntity = new OfflineInputEntity();

                modelInputEntity.Speech = stream.GetDecodeChunk();
                if (modelInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                modelInputEntity.SpeechLength = modelInputEntity.Speech.Length;
                modelInputs.Add(modelInputEntity);
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
                OfflineOutputEntity modelOutputEntity = _offlineProj.ModelProj(modelInputs);
                if (modelOutputEntity != null)
                {
                    Tensor<float>? logits_tensor = modelOutputEntity.Logits;
                    List<int[]> token_nums = new List<int[]> { };
                    List<List<int[]>> timestamps_list = new List<List<int[]>>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        int[] item = new int[logits_tensor.Dimensions[1]];
                        List<int[]> timestamps = new List<int[]>();
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                            timestamps.Add(new int[] { 0, 0 });
                        }
                        token_nums.Add(item);
                        timestamps_list.Add(timestamps);
                    }
                    token_nums = RemoveConsecutiveDuplicatesToken(token_nums);
                    int streamIndex = 0;
                    foreach (OfflineStream stream in streams)
                    {
                        stream.Tokens = token_nums[streamIndex].ToList();
                        stream.Timestamps.AddRange(timestamps_list[streamIndex]);
                        stream.RemoveDecodedChunk();
                        streamIndex++;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }
        }

        public static List<int[]> RemoveConsecutiveDuplicatesToken(List<int[]> TokensList)
        {
            var result = new List<int[]>();
            if (TokensList == null) return result;

            foreach (var tokens in TokensList)
            {
                if (tokens == null || tokens.Length == 0)
                {
                    result.Add(new int[0]); // 空序列返回空数组
                    continue;
                }

                // 第一个元素总是保留
                var decodedList = new List<int> { tokens[0] };
                for (int i = 1; i < tokens.Length; i++)
                {
                    if (tokens[i] != tokens[i - 1])
                        decodedList.Add(tokens[i]);
                }
                result.Add(decodedList.ToArray());
            }
            return result;
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
                foreach (var result in stream.Tokens.Zip<int, int[]>(stream.Timestamps))
                {
                    Int64 token = result.First;
                    int[] timestamp = result.Second;
#else
                for (int i = 0; i < stream.Tokens.Count && i < stream.Timestamps.Count; i++)
                {
                    Int64 token = stream.Tokens[i];
                    int[] timestamp = stream.Timestamps[i];
#endif
                    if (token == 2)
                    {
                        break;
                    }
                    string currText = _tokens[token].Split('\t')[0];
                    if (currText != "</s>" && currText != "<s>" && currText != "<blank>" && currText != "<unk>" && currText != "<pad>")
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
                    text_result = text_result.Replace("@@▁▁", "").Replace("▁▁ ▁▁", " ").Replace("▁▁", "").Replace("@@", " ").Replace("▁", "");
                }
                else
                {
                    text_result = text_result.Replace("▁▁▁", " ").Replace("▁▁ ▁▁", " ").Replace("▁▁", "").Replace("▁", "");
                }
                text_results.Add(text_result);
                offlineRecognizerResultEntity.Text = text_result;
                offlineRecognizerResultEntity.TextLen = text_result.Length;
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