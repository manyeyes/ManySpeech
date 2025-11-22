// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using System.Text.RegularExpressions;

namespace ManySpeech.K2TransducerAsr
{
    delegate void ForwardBatchOnline(List<OnlineStream> streams);
    public class OnlineRecognizer : IDisposable
    {
        private bool _disposed;

        private string[]? _tokens;
        private IOnlineProj? _onlineProj;

        private ForwardBatchOnline? _forwardBatch;

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, string joinerFilePath, string tokensFilePath, string configFilePath = "", string decodingMethod = "greedy_search", int sampleRate = 16000, int featureDim = 80,
            int threadsNum = 2, bool debug = false, int maxActivePaths = 4, int enableEndpoint = 0)
        {
            OnlineModel onlineModel = new OnlineModel(encoderFilePath, decoderFilePath, joinerFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            onlineModel.FeatureDim = featureDim;
            onlineModel.SampleRate = sampleRate;
            _tokens = File.ReadAllLines(tokensFilePath);

            switch (onlineModel.CustomMetadata.Model_type)
            {
                case "zipformer":
                    _onlineProj = new OnlineProjOfZipformer(onlineModel);
                    break;
                case "zipformer2":
                    _onlineProj = new OnlineProjOfZipformer2(onlineModel);
                    break;
                case "zipformer2ctc":
                    _onlineProj = new OnlineProjOfZipformer2ctc(onlineModel);
                    decodingMethod = "greedy_search_ctc";
                    break;
                case "lstm":
                    _onlineProj = new OnlineProjOfLstm(onlineModel);
                    break;
                case "conformer":
                    _onlineProj = new OnlineProjOfConformer(onlineModel);
                    break;
            }

            switch (decodingMethod)
            {
                case "greedy_search":
                    _forwardBatch = new ForwardBatchOnline(this.ForwardBatchGreedySearch);
                    break;
                case "greedy_search_ctc":
                    _forwardBatch = new ForwardBatchOnline(this.ForwardBatchGreedySearchCTC);
                    break;
                default:
                    _forwardBatch = new ForwardBatchOnline(this.ForwardBatchGreedySearch);
                    break;
            }
        }

        public OnlineStream CreateOnlineStream()
        {
            OnlineStream onlineStream = new OnlineStream(_onlineProj);
            return onlineStream;
        }

        public OnlineRecognizerResultEntity GetResult(OnlineStream stream)
        {
            OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
            List<OnlineStream> streams = new List<OnlineStream>();
            streams.Add(stream);
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = GetResults(streams);
            onlineRecognizerResultEntity = onlineRecognizerResultEntities[0];
            return onlineRecognizerResultEntity;
        }

        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            _forwardBatch.Invoke(streams);
#pragma warning restore CS8602 // 解引用可能出现空引用。
            onlineRecognizerResultEntities = this.DecodeMulti(streams);
            return onlineRecognizerResultEntities;
        }
        private void ForwardBatchGreedySearch(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            int contextSize = _onlineProj.CustomMetadata.Context_size;
            List<OnlineInputEntity> modelInputs = new List<OnlineInputEntity>();
            List<List<List<float[]>>> stateList = new List<List<List<float[]>>>();
            List<Int64[]> hypList = new List<Int64[]>();
            List<List<Int64>> tokens = new List<List<Int64>>();
            List<OnlineStream> streamsTemp = new List<OnlineStream>();
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
                stream.RemoveChunk();
                hypList.Add(stream.Hyp);
                stateList.Add(stream.States);
                tokens.Add(stream.Tokens);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            int batchSize = modelInputs.Count;
            Int64[] hyps = new Int64[contextSize * batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                Array.Copy(hypList[i].ToArray(), 0, hyps, i * contextSize, contextSize);
            }
            try
            {
                List<float[]> states = new List<float[]>();
                List<List<float[]>> stackStatesList = new List<List<float[]>>();
                stackStatesList = _onlineProj.stack_states(stateList);
                EncoderOutputEntity encoderOutput = _onlineProj.EncoderProj(modelInputs, batchSize, stackStatesList);
                int joinerDim = _onlineProj.CustomMetadata.Joiner_dim;
                int TT = encoderOutput.encoder_out.Length / joinerDim;
                DecoderOutputEntity decoderOutput = _onlineProj.DecoderProj(hyps, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;
                List<int[]> timestamp;
                int batchPerNum = TT / batchSize;

                List<int>[] timestamps = new List<int>[batchSize];
                for (int t = 0; t < batchPerNum; t++)
                {
                    float[] current_encoder_out = new float[joinerDim * batchSize];
                    for (int b = 0; b < batchSize; b++)
                    {
                        Array.Copy(encoderOutput.encoder_out, t * joinerDim + (batchPerNum * joinerDim) * b, current_encoder_out, b * joinerDim, joinerDim);
                    }
                    JoinerOutputEntity joinerOutput = _onlineProj.JoinerProj(
                        current_encoder_out, decoder_out
                    );
                    Tensor<float> logits = joinerOutput.Logits;
                    List<int[]> token_nums = new List<int[]> { };
                    int itemLength = logits.Dimensions[0] / batchSize;
                    for (int i = 0; i < batchSize; i++)
                    {
                        int[] item = new int[itemLength];
                        for (int j = i * itemLength; j < (i + 1) * itemLength; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits.Dimensions[1]; k++)
                            {
                                token_num = logits[j, token_num] > logits[j, k] ? token_num : k;
                            }
                            item[j - i * itemLength] = (int)token_num;

                        }
                        token_nums.Add(item);
                    }
                    bool emitted = false;
                    for (int m = 0; m < token_nums.Count; m++)
                    {
                        int y = token_nums[m][0];
                        if (tokens[m] == null)
                        {
                            tokens[m] = new List<Int64>();
                        }
                        if (timestamps[m] == null)
                        {
                            timestamps[m] = new List<int>();
                        }
                        if (y != _onlineProj.Blank_id && y != _onlineProj.Unk_id && y != 1)
                        {
                            tokens[m].Add(y);
                            timestamps[m].Add(t);
                            emitted = true;
                        }
                        else
                        {
                            //do nothing 
                        }
                    }
                    if (emitted)
                    {
                        Int64[] decoder_input = new Int64[contextSize * batchSize];
                        for (int m = 0; m < batchSize; m++)
                        {
                            Array.Copy(tokens[m].ToArray(), tokens[m].Count - contextSize, decoder_input, m * contextSize, contextSize);
                        }
                        decoder_out = _onlineProj.DecoderProj(decoder_input, batchSize).decoder_out;
                    }

                }
                List<List<List<float[]>>> next_statesList = new List<List<List<float[]>>>();
                next_statesList = _onlineProj.unstack_states(encoderOutput.encoder_out_states);
                int streamIndex = 0;
                foreach (OnlineStream stream in streams)
                {
                    Array.Copy(tokens[streamIndex].ToArray(), tokens[streamIndex].Count - contextSize, stream.Hyp, 0, contextSize);
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps.AddRange(timestamps[streamIndex]);
                    stream.States = next_statesList[streamIndex];
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Online recognition failed", ex);
            }
        }
        private void ForwardBatchGreedySearchCTC(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            int contextSize = _onlineProj.CustomMetadata.Context_size;
            List<OnlineInputEntity> modelInputs = new List<OnlineInputEntity>();
            List<List<List<float[]>>> stateList = new List<List<List<float[]>>>();
            List<List<Int64>> tokens = new List<List<Int64>>();
            List<List<int>> timestamps = new List<List<int>>();
            List<OnlineStream> streamsTemp = new List<OnlineStream>();
            List<int> numTrailingBlanks = new List<int>();
            List<int> frameOffsets = new List<int>();
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
                stream.RemoveChunk();
                stateList.Add(stream.States);
                tokens.Add(stream.Tokens);
                numTrailingBlanks.Add(stream.NumTrailingBlank);
                frameOffsets.Add(stream.FrameOffset);
                timestamps.Add(stream.Timestamps);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            int batchSize = modelInputs.Count;
            try
            {
                List<float[]> states = new List<float[]>();
                List<List<float[]>> stackStatesList = new List<List<float[]>>();
                stackStatesList = _onlineProj.stack_states(stateList);
                EncoderOutputEntity encoderOutput = _onlineProj.EncoderProj(modelInputs, batchSize, stackStatesList);
                //ctc decode
                float[] log_probs = encoderOutput.encoder_out;
                int vocab_size = _tokens.Length;
                int num_frames = log_probs.Length / batchSize / vocab_size;
                int pIndex = 0;
                for (int b = 0; b < batchSize; ++b)
                {
                    float[] log_probs_b = new float[num_frames * vocab_size];
                    Array.Copy(log_probs, b * num_frames * vocab_size, log_probs_b, 0, num_frames * vocab_size);
                    Int64 prev_id = -1;
                    for (int t = 0; t < num_frames; ++t, pIndex += vocab_size)
                    {
                        Int64 y = Array.IndexOf(log_probs_b, log_probs_b.Skip(pIndex).Take(vocab_size).Max(), pIndex);
                        y = y - pIndex;
                        if (y == _onlineProj.Blank_id)
                        {
                            numTrailingBlanks[b] += 1;
                        }
                        else
                        {
                            numTrailingBlanks[b] = 0;
                        }

                        if (y != _onlineProj.Blank_id && y != prev_id)
                        {
                            tokens[b].Add(y);
                            timestamps[b].Add(t + frameOffsets[b]);
                        }
                        prev_id = y;
                    }
                    pIndex = 0;
                }
                // Update frame_offset
                for (int b = 0; b < batchSize; ++b)
                {
                    frameOffsets[b] += num_frames;
                }
                List<List<List<float[]>>> next_statesList = new List<List<List<float[]>>>();
                next_statesList = _onlineProj.unstack_states(encoderOutput.encoder_out_states);
                int streamIndex = 0;
                foreach (OnlineStream stream in streams)
                {
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps = timestamps[streamIndex];
                    stream.States = next_statesList[streamIndex];
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Online recognition failed", ex);
            }
        }

        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OnlineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    string currToken = _tokens[token].Split(' ')[0];
                    if (currToken != "<blk>" && currToken != "<sos/eos>" && currToken != "<unk>")
                    {
                        if (IsChinese(_tokens[token], true))
                        {
                            text_result += currToken;
                        }
                        else
                        {
                            text_result += currToken + "";
                        }
                    }
                }
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                onlineRecognizerResultEntity.Text = CheckText(text_result.Replace("▁", " ")).ToLower();
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return onlineRecognizerResultEntities;
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
        private static string CheckTextFirst(string str)
        {
            int[] intArray = new int[str.Length];
            for (int i = 0; i < str.Length; i++)
            {
                // 将字符转换为对应的整数值
                intArray[i] = (int)str[i];
            }
            //
            string[] hexArray = new string[intArray.Length];
            for (int i = 0; i < intArray.Length; i++)
            {
                hexArray[i] = $"<0x{intArray[i]:X2}>";
            }
            foreach (string hex in hexArray)
            {
                Console.Write(hex + " ");
            }
            return string.Join("", hexArray);
        }

        private static string CheckText(string text)
        {
            Regex r = new Regex(@"\<(\w+)\>");
            var matches = r.Matches(text);
            if (matches.Count == 0)
            {
                text = Utils.ByteDataHelper.SmartByteDecode(text);
            }
            int mIndex = -1;
            List<string> hexsList = new List<string>();
            List<string> strsList = new List<string>();
            StringBuilder hexSB = new StringBuilder();
            foreach (var m in matches.Cast<Match>().ToArray())
            {
                if (mIndex == -1)
                {
                    hexSB.Append(m.Groups[0].ToString());
                }
                else
                {
                    if (m.Index - mIndex == 6)
                    {
                        hexSB.Append(m.Groups[0].ToString());
                    }
                    else
                    {
                        hexsList.Add(hexSB.ToString());
                        strsList.Add(hexSB.ToString().Replace("<0x", "").Replace(">", ""));
                        hexSB = new StringBuilder();
                        hexSB.Append(m.Groups[0].ToString());
                    }
                }
                if (m == matches.Cast<Match>().Last())
                {
                    hexsList.Add(hexSB.ToString());
                    strsList.Add(hexSB.ToString().Replace("<0x", "").Replace(">", ""));
                }
                mIndex = m.Index;
            }
#if NET6_0_OR_GREATER || NETCOREAPP3_1_OR_GREATER
            // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
            foreach (var item in hexsList.Zip<string, string>(strsList))
            {
                text = text.Replace(item.First, HexToStr(item.Second));
            }
#else
            // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
            for (int i = 0; i < hexsList.Count && i < strsList.Count; i++)
            {
                text = text.Replace(hexsList[i], HexToStr(strsList[i]));
            }
#endif
            return text;
        }

        /// <summary>
        /// 从16进制转换成汉字
        /// </summary>
        /// <param name="hex"></param>
        /// <returns></returns>
        public static string HexToStr(string hex)
        {
            if (hex == null)
                throw new ArgumentNullException("hex");
            if (hex.Length % 2 != 0)
            {
                hex += "20";//空格
            }
            // 需要将 hex 转换成 byte 数组。
            byte[] bytes = new byte[hex.Length / 2];
            for (int i = 0; i < bytes.Length; i++)
            {
                try
                {
                    // 每两个字符是一个 byte。
                    bytes[i] = byte.Parse(hex.Substring(i * 2, 2),
                        System.Globalization.NumberStyles.HexNumber);
                }
                catch
                {
                    throw new ArgumentException("hex is not a valid hex number!", "hex");
                }
            }
            string str = Encoding.GetEncoding("utf-8").GetString(bytes);
            return str;
        }

        public void DisposeOnlineStream(OnlineStream onlineStream)
        {
            if (onlineStream != null)
            {
                onlineStream.Dispose();
            }
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
                    if (_tokens != null)
                    {
                        _tokens = null;
                    }
                    if (_forwardBatch != null)
                    {
                        _forwardBatch = null;
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