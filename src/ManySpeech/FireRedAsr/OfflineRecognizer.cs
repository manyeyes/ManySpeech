// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.FireRedAsr.Model;
using System.Text.RegularExpressions;

namespace ManySpeech.FireRedAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2025 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private string _mvnFilePath;
        private IAsrProj _asrProj;

        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string mvnFilePath, string tokensFilePath, string configFilePath = "", string ctcFilePath = "", int threadsNum = 1)
        {
            _mvnFilePath = mvnFilePath;
            AsrModel asrModel = new AsrModel(encoderFilePath, decoderFilePath, ctcFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _asrProj = new AsrProjOfAED(asrModel);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_mvnFilePath, _asrProj);
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
            //this._logger.LogInformation("get features begin");
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
            List<AsrInputEntity> modelInputs = new List<AsrInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            //List<Int64[]> hypList = new List<Int64[]>();
            List<List<Int64>> tokensList = new List<List<Int64>>();
            List<List<int[]>> timestampsList = new List<List<int[]>>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                AsrInputEntity asrInputEntity = new AsrInputEntity();

                asrInputEntity.Speech = stream.GetDecodeChunk();
                if (asrInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                asrInputEntity.SpeechLength = asrInputEntity.Speech.Length;
                modelInputs.Add(asrInputEntity);
                //hypList.Add(stream.Hyp);
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
                stackStatesList = _asrProj.stack_states(statesList);
                EncoderOutputEntity encoderOutputEntity = _asrProj.EncoderProj(modelInputs);
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
                if (_asrProj.DecoderSession != null)
                {
                    for (int i = 0; i < maxlen; i++)
                    {
                        DecoderOutputEntity decoderOutputEntity = _asrProj.DecoderProj(tokensList, encoder_outputs, src_mask, stackStatesList);
                        // 合并对应索引的子列表
                        tokensList = tokensList.Zip(decoderOutputEntity.TokensList, (a, b) => a.Concat(b).ToList()).ToList();
                        stackStatesList = decoderOutputEntity.CacheList;
                        bool allEnd = tokensList.All(item => item.Any() && item.Last() == _asrProj.Eos_id);
                        if (allEnd)
                        {
                            break;
                        }
                    }
                }
                for (int i = 0; i < batchSize; i++)
                {
                    List<int> tokens = tokensList[i].Select(x => (int)x).ToList();
                    tokens = tokens.Skip(1).ToList();
                    // 找到第一个 Eos_id 的索引
                    int eosIndex = tokens.FindIndex(t => t == _asrProj.Eos_id) + 1;
                    // 如果找到 Eos_id，移除它及其后面的所有元素
                    if (eosIndex != -1)
                    {
                        tokens.RemoveRange(eosIndex, tokens.Count - eosIndex);
                    }
                    if (_asrProj.CtcSession != null)
                    {
                        double frameShift = 0.04; // 40ms
                        CtcOutputEntity ctcOutputEntity = _asrProj.CtcProj(encoder_outputs, batchSize: batchSize);
                        List<float[]> ctcLogits = ctcOutputEntity.LogitsList[i];
                        if (tokens.Count == 0)
                        {
                            int t = ctcLogits.Count;
                            int[] item = new int[ctcLogits.Count];
                            for (int j = 0; j < ctcLogits.Count; j++)
                            {
                                if (j > t - t / 30 && ctcLogits[j].Average() / ctcLogits[j][0] > 100000)
                                {
                                    ctcLogits[j][0] = ctcLogits[j][0] < -0.000001f && ctcLogits[j][0] > -0.0001f ? ctcLogits[j][0] * 10000000 : ctcLogits[j][0];
                                }
                                int token_num = 0;
                                for (int k = 1; k < ctcLogits[j].Length; k++)
                                {
                                    token_num = ctcLogits[j][token_num] > ctcLogits[j][k] ? token_num : k;
                                }
                                item[j] = (int)token_num;
                            }
                            tokens = RemoveDuplicatesAndBlank(item);
                            tokensList[i].AddRange(tokens.Select(x => (Int64)x).ToArray());
                        }
                        (List<double> startTimes, List<double> endTimes) = GetCtcTimestamp(ctcLogits, tokens, blankId: _asrProj.Blank_id, frameShift: frameShift);
                        timestampsList.Add(ConvertToMilliseconds(startTimes, endTimes));
                    }
                    else
                    {
                        List<int[]> timestamps = new List<int[]>();
                        for (int j = 0; j < tokens.Count; j++)
                        {
                            timestamps.Add(new int[2]);
                        }
                        timestampsList.Add(timestamps);
                    }
                }
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    stream.Tokens = tokensList[streamIndex].ToList();
                    stream.Timestamps.AddRange(timestampsList[streamIndex]);
                    stream.RemoveDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }

        }

        /// <summary>
        /// Removes duplicate tokens and blank tokens from the sequence
        /// </summary>
        /// <param name="yseq">Original token sequence</param>
        /// <param name="blank_id">The identifier for blank token (default: 0)</param>
        /// <returns>List of tokens with duplicates and blank tokens removed</returns>
        public List<int> RemoveDuplicatesAndBlank(int[] yseq, int blank_id = 0)
        {
            // Null and empty check to avoid exceptions caused by null array
            if (yseq == null || yseq.Length == 0)
            {
                return new List<int>();
            }

            int prev_token = -1;
            List<int> decoded_tokens = new List<int>();

            // Iterate through the token sequence to remove duplicates and blank tokens
            foreach (int token in yseq)
            {
                // Only keep: current token != previous token AND current token != blank token
                if (token != prev_token && token != blank_id)
                {
                    decoded_tokens.Add(token);
                }
                prev_token = token;
            }

            return decoded_tokens;
        }

        public static List<int[]> ConvertToMilliseconds(List<double> startTimes, List<double> endTimes)
        {
            if (startTimes.Count != endTimes.Count)
                throw new ArgumentException("The length of the start and end time lists must be the same.");

            return startTimes.Zip(endTimes, (s, e) => new[]
            {
                (int)Math.Round(s * 1000, MidpointRounding.AwayFromZero),
                (int)Math.Round(e * 1000, MidpointRounding.AwayFromZero)
            }).ToList();
        }

        /// <summary>
        /// Get the timestamp of each token from CTC logits (input in List<float[]> format)
        /// </summary>
        /// <param name="logProbs">Logits output by the model (after log_softmax calculation), where the List length is T and each float[] length is C (including blank token)</param>
        /// <param name="tokens">Target token ID list (excluding blank token)</param>
        /// <param name="blankId">Blank token index</param>
        /// <param name="frameShift">Frame shift in seconds</param>
        /// <returns>
        /// A tuple (startTimes, endTimes) where each element represents the start and end time (in seconds) of the corresponding token.
        /// Returns (null, null) if alignment fails or input is invalid.
        /// </returns>
        public static (List<double> startTimes, List<double> endTimes) GetCtcTimestamp(
            List<float[]> logProbs,
            List<int> tokens,
            int blankId,
            double frameShift)
        {
            if (tokens == null || tokens.Count == 0)
                return (null, null);

            int T = logProbs.Count;
            if (tokens.Count > T)
            {
                // Log warning: Token sequence length exceeds number of frames
                return (null, null);
            }

            // Perform forced alignment to map tokens to time frames
            int[] alignment;
            try
            {
                alignment = Utils.ComputeHelper.ForcedAlign(logProbs, tokens.ToArray(), blankId);
            }
            catch (Exception)
            {
                // Log warning: Forced alignment failed
                return (null, null);
            }

            // Convert alignment results (frame-level token mapping) to timestamps
            List<double> startTimes = new List<double>();
            List<double> endTimes = new List<double>();

            int prevToken = blankId;
            for (int t = 0; t < alignment.Length; t++)
            {
                int token = alignment[t];
                if (token != blankId) // Current frame has a non-blank token
                {
                    // New token starts (different from previous token)
                    if (token != prevToken)
                    {
                        if (prevToken != blankId)
                            endTimes.Add(t * frameShift);
                        startTimes.Add(t * frameShift);
                        prevToken = token;
                    }
                }
                else // Current frame is blank token
                {
                    if (prevToken != blankId)
                    {
                        endTimes.Add(t * frameShift);
                        prevToken = blankId;
                    }
                }
            }

            // Last token hasn't been closed (ends at the last frame)
            if (prevToken != blankId)
                endTimes.Add(alignment.Length * frameShift);

            // Validation: Ensure the number of timestamp segments matches the number of tokens
            if (startTimes.Count != tokens.Count)
            {
                // Log warning: Mismatch between alignment segments and token count
                return (null, null);
            }

            return (startTimes, endTimes);
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
                foreach (var result in stream.Tokens.Zip<Int64, int[]>(stream.Timestamps))
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
                    string currText = _tokens[token].Split(new char[] { '\t', ' ' })[0].ToLower();
                    if (currText != "</s>" && currText != "<s>" && currText != "<sos/eos>" && currText != "<blank>" && currText != "<unk>" && currText != "<sos>" && currText != "<eos>" && currText != "<pad>")
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
                    if (_asrProj != null)
                    {
                        _asrProj.Dispose();
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