// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.K2TransducerAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;
using System.Text.RegularExpressions;

namespace ManySpeech.K2TransducerAsr
{
    public delegate void ForwardOffline(OfflineStream stream);
    public delegate void ForwardBatchOffline(List<OfflineStream> streams);
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private WavFrontend _wavFrontend;
        private FrontendConfEntity _frontendConfEntity;
        private string[] _tokens;
        private int _max_sym_per_frame = 1;
        private int _blank_id = 0;
        private int _unk_id = 2;

        private OfflineModel _offlineModel;
        private ForwardBatchOffline? _forwardBatch;
        private ForwardOffline? _forward;
        private IOfflineProj? _offlineProj;
        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string joinerFilePath, string tokensFilePath,
            string decodingMethod = "greedy_search", int sampleRate = 16000, int featureDim = 80, int threadsNum = 2, bool debug = false)
        {
            _offlineModel = new OfflineModel(encoderFilePath, decoderFilePath, joinerFilePath, threadsNum);
            _offlineModel.FeatureDim = featureDim;
            _tokens = File.ReadAllLines(tokensFilePath);

            _frontendConfEntity = new FrontendConfEntity();
            _frontendConfEntity.fs = sampleRate;
            _frontendConfEntity.n_mels = featureDim;
            _wavFrontend = new WavFrontend(_frontendConfEntity);
            switch (_offlineModel.CustomMetadata.Model_type)
            {
                case "zipformer":
                case "zipformer2":
                case "conformer":
                case "lstm":
                    _offlineProj = new OfflineProjOfTransducer(_offlineModel);
                    break;
                case "zipformer2ctc":
                    _offlineProj = new OfflineProjOfZipformer2ctc(_offlineModel);
                    decodingMethod = "greedy_search_ctc";
                    break;
                default:
                    _offlineProj = new OfflineProjOfTransducer(_offlineModel);
                    break;
            }
            switch (decodingMethod)
            {
                case "greedy_search":
                    _forward = new ForwardOffline(this.ForwardGreedySearch);
                    _forwardBatch = new ForwardBatchOffline(this.ForwardBatchGreedySearch);
                    break;
                case "greedy_search_ctc":
                    _forward = new ForwardOffline(this.ForwardGreedySearchCTC);
                    _forwardBatch = new ForwardBatchOffline(this.ForwardBatchGreedySearchCTC);
                    break;
                default:
                    _forward = new ForwardOffline(this.ForwardGreedySearch);
                    _forwardBatch = new ForwardBatchOffline(this.ForwardBatchGreedySearch);
                    break;
            }
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream offlineStream = new OfflineStream(_offlineModel.CustomMetadata, sampleRate: _frontendConfEntity.fs, featureDim: _frontendConfEntity.n_mels);
            return offlineStream;
        }

        public OfflineRecognizerResultEntity GetResult(OfflineStream stream)
        {
            OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
            _forward.Invoke(stream);
            offlineRecognizerResultEntity = this.DecodeMulti(new List<OfflineStream>() { stream })[0];
            return offlineRecognizerResultEntity;
        }

        public List<OfflineRecognizerResultEntity> GetResults(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
            _forwardBatch.Invoke(streams);
            offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private void ForwardGreedySearch(OfflineStream stream)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            int batchSize = 1;
            OfflineRecognizerResultEntity modelOutput = new OfflineRecognizerResultEntity();
            try
            {
                List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
                modelInputs.Add(stream.OfflineInputEntity);
                EncoderOutputEntity encoderOutput = _offlineProj.EncoderProj(modelInputs, batchSize);
                int TT = encoderOutput.encoder_out.Length / 512;
                int t = 0;
                Int64[] hyp = new Int64[] { -1, _blank_id };
                Int64[] hyps = new Int64[contextSize * batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    Array.Copy(hyp, 0, hyps, i * contextSize, contextSize);
                }
                DecoderOutputEntity decoderOutput = _offlineProj.DecoderProj(hyps, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;


                List<Int64> hypList = new List<Int64>();
                hypList.Add(-1);
                hypList.Add(_blank_id);
                // timestamp[i] is the frame index after subsampling
                // on which hyp[i] is decoded
                List<int> timestamp = new List<int>();
                // Maximum symbols per utterance.
                int max_sym_per_utt = 1000;
                // symbols per frame
                int sym_per_frame = 0;
                // symbols per utterance decoded so far
                int sym_per_utt = 0;
                while (t < TT && sym_per_utt < max_sym_per_utt)
                {
                    if (sym_per_frame >= _max_sym_per_frame)
                    {
                        sym_per_frame = 0;
                        t += 1;
                        continue;
                    }
                    // fmt: off
                    float[] current_encoder_out = new float[512];
                    Array.Copy(encoderOutput.encoder_out, t * 512, current_encoder_out, 0, 512);
                    // fmt: on
                    JoinerOutputEntity joinerOutput = _offlineProj.JoinerProj(
                        current_encoder_out, decoder_out
                    );
                    Tensor<float> logits = joinerOutput.Logits;
                    List<int[]> token_nums = new List<int[]> { };
                    int itemLength = logits.Dimensions[0];
                    for (int i = 0; i < 1; i++)
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
                    int y = token_nums[0][0];
                    if (y != _blank_id && y != _unk_id)
                    {
                        hypList.Add(y);
                        timestamp.Add(t);
                        Int64[] decoder_input = new Int64[contextSize * batchSize];
                        for (int m = 0; m < batchSize; m++)
                        {
                            Array.Copy(hypList.ToArray(), hypList.Count - contextSize, decoder_input, m * contextSize, contextSize);
                        }
                        decoder_out = _offlineProj.DecoderProj(decoder_input, batchSize).decoder_out;
                        sym_per_utt += 1;
                        sym_per_frame += 1;
                    }
                    else
                    {
                        sym_per_frame = 0;
                        t += 1;
                    }
                }
                stream.Tokens = hypList;
                stream.Timestamps.AddRange(timestamp);
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }
        }

        private void ForwardBatchGreedySearch(List<OfflineStream> streams)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            int batchSize = streams.Count;
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            foreach (OfflineStream stream in streams)
            {
                modelInputs.Add(stream.OfflineInputEntity);
            }
            try
            {
                EncoderOutputEntity encoderOutput = _offlineProj.EncoderProj(modelInputs, batchSize);
                int TT = encoderOutput.encoder_out.Length / 512;
                Int64[] hyp = new Int64[] { -1, _blank_id };
                Int64[] hyps = new Int64[contextSize * batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    Array.Copy(hyp, 0, hyps, i * contextSize, contextSize);
                }
                DecoderOutputEntity decoderOutput = _offlineProj.DecoderProj(hyps, batchSize);
                float[] decoder_out = decoderOutput.decoder_out;
                // timestamp[i] is the frame index after subsampling
                // on which hyp[i] is decoded
                List<int[]> timestamp;
                int batchPerNum = TT / batchSize;
                List<Int64>[] tokens = new List<Int64>[batchSize];
                List<int>[] timestamps = new List<int>[batchSize];
                for (int t = 0; t < batchPerNum; t++)
                {
                    // fmt: off
                    float[] current_encoder_out = new float[512 * batchSize];
                    for (int b = 0; b < batchSize; b++)
                    {
                        Array.Copy(encoderOutput.encoder_out, t * 512 + (batchPerNum * 512) * b, current_encoder_out, b * 512, 512);
                    }
                    // fmt: on
                    JoinerOutputEntity joinerOutput = _offlineProj.JoinerProj(
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
                            for (int i = 0; i < batchSize; i++)
                            {
                                tokens[m].Add(_blank_id);
                                tokens[m].Add(_blank_id);
                            }
                        }
                        if (timestamps[m] == null)
                        {
                            timestamps[m] = new List<int>();
                            for (int i = 0; i < batchSize; i++)
                            {
                                timestamps[m].Add(0);
                                timestamps[m].Add(0);
                            }
                        }
                        if (y != _blank_id && y != _unk_id)
                        {
                            tokens[m].Add(y);
                            timestamps[m].Add(t);
                            emitted = true;
                        }
                        else
                        {
                        }
                    }
                    if (emitted)
                    {
                        Int64[] decoder_input = new Int64[contextSize * batchSize];
                        for (int m = 0; m < batchSize; m++)
                        {
                            Array.Copy(tokens[m].ToArray(), tokens[m].Count - contextSize, decoder_input, m * contextSize, contextSize);
                        }
                        decoder_out = _offlineProj.DecoderProj(decoder_input, batchSize).decoder_out;
                    }

                }
                int streamIndex = 0;
                foreach (OfflineStream stream in streams)
                {
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps.AddRange(timestamps[streamIndex]);
                    stream.RemoveSamples();
                    streamIndex++;
                }

            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }
        }

        private void ForwardGreedySearchCTC(OfflineStream stream)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            int batchSize = 1;
            OfflineRecognizerResultEntity modelOutput = new OfflineRecognizerResultEntity();
            try
            {
                List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
                modelInputs.Add(stream.OfflineInputEntity);
                EncoderOutputEntity encoderOutput = _offlineProj.EncoderProj(modelInputs, batchSize);
                //ctc decode
                int numTrailingBlank = 0;
                int frameOffset = 0;
                List<Int64> tokens = new List<Int64>();
                tokens.Add(-1);
                tokens.Add(_blank_id);
                // timestamp[i] is the frame index after subsampling
                List<int> timestamp = new List<int>();

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
                        if (y == _offlineProj.Blank_id)
                        {
                            numTrailingBlank += 1;
                        }
                        else
                        {
                            numTrailingBlank = 0;
                        }

                        if (y != _offlineProj.Blank_id && y != prev_id)
                        {
                            tokens.Add(y);
                            timestamp.Add(t + frameOffset);
                        }
                        prev_id = y;
                    }
                    pIndex = 0;
                }
                stream.NumTrailingBlank = numTrailingBlank;
                stream.Tokens = tokens;
                stream.Timestamps.AddRange(timestamp);
            }
            catch (Exception ex)
            {
                //
            }
        }

        private void ForwardBatchGreedySearchCTC(List<OfflineStream> streams)
        {
            int contextSize = _offlineModel.CustomMetadata.Context_size;
            int batchSize = streams.Count;
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            try
            {
                List<List<Int64>> tokens = new List<List<Int64>>();
                List<List<int>> timestamps = new List<List<int>>();
                List<int> numTrailingBlanks = new List<int>();
                List<int> frameOffsets = new List<int>();
                foreach (OfflineStream stream in streams)
                {
                    modelInputs.Add(stream.OfflineInputEntity);
                    numTrailingBlanks.Add(stream.NumTrailingBlank);
                    frameOffsets.Add(stream.FrameOffset);
                    tokens.Add(stream.Tokens);
                    timestamps.Add(stream.Timestamps);
                }
                EncoderOutputEntity encoderOutput = _offlineProj.EncoderProj(modelInputs, batchSize);
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
                        if (y == _offlineProj.Blank_id)
                        {
                            numTrailingBlanks[b] += 1;
                        }
                        else
                        {
                            numTrailingBlanks[b] = 0;
                        }

                        if (y != _offlineProj.Blank_id && y != prev_id)
                        {
                            tokens[b].Add(y);
                            timestamps[b].Add(t + frameOffsets[b]);
                        }
                        prev_id = y;
                    }
                    pIndex = 0;
                }
                int streamIndex = 0;
                foreach (OfflineStream stream in streams)
                {
                    stream.Tokens = tokens[streamIndex];
                    stream.Timestamps = timestamps[streamIndex];
                    stream.NumTrailingBlank = numTrailingBlanks[streamIndex];
                    stream.RemoveSamples();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Speech recognition failed", ex);
            }
        }

        private List<OfflineRecognizerResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineRecognizerResultEntity> offlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (token == -1)
                    {
                        continue;
                    }
                    if (_tokens[token].Split(' ')[0] != "<blk>" && _tokens[token].Split(' ')[0] != "<sos/eos>" && _tokens[token].Split(' ')[0] != "<unk>")
                    {
                        if (IsChinese(_tokens[token], true))
                        {
                            text_result += _tokens[token].Split(' ')[0];
                        }
                        else
                        {
                            text_result += _tokens[token].Split(' ')[0] + "";
                        }
                    }
                }
                OfflineRecognizerResultEntity offlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                offlineRecognizerResultEntity.Text = CheckText(text_result.Replace("▁", " ")).ToLower();
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

        private static string CheckText(string text)
        {
            Regex r = new Regex(@"\<(\w+)\>");
            var matches = r.Matches(text);
            if (matches.Count == 0)
            {
                text = Utils.ByteDataHelper.SmartByteDecode(text.Replace(" ", ""));
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
#if NET6_0_OR_GREATER
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
                    if (_forwardBatch != null)
                    {
                        _forwardBatch = null;
                    }
                    if (_forward != null)
                    {
                        _forward = null;
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