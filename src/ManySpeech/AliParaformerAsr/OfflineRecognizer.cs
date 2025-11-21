// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.RegularExpressions;

namespace ManySpeech.AliParaformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OfflineRecognizer : IDisposable
    {
        private bool _disposed;

        private OfflineModel _offlineModel;
        private string[] _tokens;
        private ConfEntity _confEntity;
        private string _mvnFilePath;
        private IOfflineProj _offlineProj;

        public OfflineRecognizer(string modelFilePath, string configFilePath, string mvnFilePath, string tokensFilePath, string modelebFilePath = "", string hotwordFilePath = "", int batchSize = 1, int threadsNum = 1)
        {
            _offlineModel = new OfflineModel(modelFilePath: modelFilePath, modelebFilePath: modelebFilePath, threadsNum);
            _confEntity = LoadConf(configFilePath);
            _offlineModel.Use_itn = _confEntity.use_itn;
            _mvnFilePath = mvnFilePath;
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
            if (_tokens == null || _tokens.Length == 0)
            {
                throw new Exception("tokens invalid");
            }
            if (!string.IsNullOrEmpty(hotwordFilePath))
            {
                List<int[]>? hotwords = GetHotwords(_tokens, hotwordFilePath);
                _offlineModel.Hotwords = hotwords;
            }
            switch (_confEntity.model.ToLower())
            {
                case "paraformer":
                    _offlineProj = new OfflineProjOfParaformer(_offlineModel);
                    break;
                case "sensevoicesmall":
                    _offlineProj = new OfflineProjOfSenseVoiceSmall(_offlineModel);
                    break;
                case "seacoparaformer":
                    _offlineProj = new OfflineProjOfSeacoParaformer(_offlineModel);
                    break;
                default:
                    _offlineProj = new OfflineProjOfParaformer(_offlineModel);
                    break;
            }
        }
        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confJsonEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confJsonEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
                    confJsonEntity = Utils.PreloadHelper.ReadJson(configFilePath); // To compile for AOT
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confJsonEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
                }
            }
            return confJsonEntity;
        }
        private List<int[]>? GetHotwords(string[] tokens, string hotwordFilePath)
        {
            List<int[]>? hotwords = new List<int[]>();
            if (File.Exists(hotwordFilePath))
            {
                string[] sentences = File.ReadAllLines(hotwordFilePath);
                foreach (string sentence in sentences)
                {
                    string[] wordList = new string[] { sentence };//TODO:分词
                    foreach (string word in wordList)
                    {
                        List<int> ids = word.ToCharArray().Select(x => Array.IndexOf(tokens, x.ToString())).Where(x => x != -1).ToList();
                        hotwords.Add(ids.ToArray());
                    }
                }
                hotwords.Add(new int[] { _offlineModel.Sos_eos_id });
            }
            return hotwords;
        }

        public OfflineStream CreateOfflineStream()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException("OfflineRecognizer");
            }
            OfflineStream offlineStream = new OfflineStream(_mvnFilePath, _confEntity);
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
            List<OfflineInputEntity> modelInputs = new List<OfflineInputEntity>();
            foreach (OfflineStream stream in streams)
            {
                var decodeChunk = stream.GetDecodeChunk();
                modelInputs.Add(decodeChunk);
            }
            try
            {
                ModelOutputEntity modelOutputEntity = _offlineProj.ModelProj(modelInputs);
                if (modelOutputEntity != null)
                {
                    Tensor<float>? logits_tensor = modelOutputEntity.model_out;
                    int[] token_nums_tensor = modelOutputEntity.model_out_lens;
                    List<Int64[]> token_nums = new List<Int64[]> { };
                    List<List<int[]>> timestamps_list = new List<List<int[]>>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[logits_tensor.Dimensions[1]];
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
                        //List<Int64> item1 = new List<Int64>(item);
                        //item1.Remove(item1.First());
                        //List<Int64> item2 = new List<Int64>(item);
                        //item2.RemoveAt(item2.Count - 1);
                        //List<Int64> newItem = new List<Int64>();
                        //int itemIndex = 0;
                        //foreach (var itemTemp in item1.Zip<Int64, Int64>(item2))
                        //{
                        //    if (itemTemp.First != itemTemp.Second)
                        //    {
                        //        newItem.Add(item[itemIndex]);
                        //    }
                        //    itemIndex++;
                        //}
                        //newItem.Add(item.Last());
                        //item = newItem.ToArray();
                        token_nums.Add(item);
                        timestamps_list.Add(timestamps);
                    }
                    if (modelOutputEntity.cif_peak_tensor != null)
                    {
                        timestamps_list = new List<List<int[]>>();
                        Tensor<float> cif_peak_tensor = modelOutputEntity.cif_peak_tensor;
                        for (int i = 0; i < cif_peak_tensor.Dimensions[0]; i++)
                        {
                            float[] us_cif_peak = new float[cif_peak_tensor.Dimensions[1]];
                            Array.Copy(cif_peak_tensor.ToArray(), i * us_cif_peak.Length, us_cif_peak, 0, us_cif_peak.Length);
                            List<int[]> timestamps = time_stamp_lfr6_onnx(us_cif_peak, token_nums[i]);
                            timestamps_list.Add(timestamps);
                        }
                    }
                    int streamIndex = 0;
                    foreach (OfflineStream stream in streams)
                    {
                        stream.Tokens = token_nums[streamIndex].ToList();
                        stream.Timestamps.AddRange(timestamps_list[streamIndex]);
                        stream.RemoveChunk();
                        streamIndex++;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline recognition failed", ex);
            }
        }

        private List<int[]> time_stamp_lfr6_onnx(float[] us_cif_peak, long[] tokens, float begin_time = 0.0F, float total_offset = -1.5F)
        {
            List<float[]> timestamp_list = new List<float[]>();
            int START_END_THRESHOLD = 5;
            int MAX_TOKEN_DURATION = 30;
            float TIME_RATE = 10.0F * 6 / 1000 / 3;//3 times upsampled

            int num_frames = us_cif_peak.Length;
            if (tokens.Last() == 2)
            {
                long[] newTokens = new long[tokens.Length - 1];
                Array.Copy(tokens, 0, newTokens, 0, newTokens.Length);
                tokens = newTokens;
            }
            List<float> fire_place = new List<float>();
            for (int i = 0; i < us_cif_peak.Length; i++)
            {
                if (us_cif_peak[i] > 1.0F - 1e-4)
                {
                    fire_place.Add(i + total_offset);
                }
            }
            List<bool> new_char_list = new List<bool>();
            //begin silence
            if (fire_place[0] > START_END_THRESHOLD)
            {
                timestamp_list.Add(new float[] { 0.0f, fire_place[0] * TIME_RATE });
                new_char_list.Add(false);
            }
            // tokens timestamp
            List<int[]> timestamps = new List<int[]>();
            for (int i = 0; i < fire_place.Count - 1; i++)
            {
                if (tokens[i] == 1)
                {
                    new_char_list.Add(false);
                }
                else
                {
                    new_char_list.Add(true);
                }
                if (i == fire_place.Count - 2 || MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION)
                {
                    timestamp_list.Add(new float[] { fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE });
                }
                else
                {
                    float _split = fire_place[i] + MAX_TOKEN_DURATION;
                    timestamp_list.Add(new float[] { fire_place[i] * TIME_RATE, _split * TIME_RATE });
                    timestamp_list.Add(new float[] { _split * TIME_RATE, fire_place[i + 1] * TIME_RATE });
                    new_char_list.Add(false);
                }
            }
            // tail token and end silence
            if (num_frames - fire_place.Last() > START_END_THRESHOLD)
            {
                float _end = (float)((num_frames + fire_place.Last()) / 2);
                timestamp_list.Last()[1] = _end * TIME_RATE;
                timestamp_list.Add(new float[] { _end * TIME_RATE, num_frames * TIME_RATE });
                new_char_list.Add(false);
            }
            else
            {
                timestamp_list.Last()[1] = num_frames * TIME_RATE;
            }
            if (begin_time > 0.0F)
            {
                for (int i = 0; i < timestamp_list.Count; i++)
                {
                    timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0F;
                    timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0F;
                }
            }
            new_char_list.Add(true);
#if NET6_0_OR_GREATER
            // .NET 6.0及更高版本：使用泛型Zip写法（保留原逻辑）
            foreach (var item in new_char_list.Zip<bool, float[]>(timestamp_list))
            {
                bool charX = item.First;
                float[] timestamp = item.Second;
                if (charX)
                {
                    timestamps.Add(new int[] { (int)(timestamp[0] * 1000), (int)(timestamp[1] * 1000) });
                }
            }
#else
            // 低版本框架（如.NET Standard 2.0）：使用兼容的Zip重载
            for (int i = 0; i < new_char_list.Count && i < timestamp_list.Count; i++)
            {
                bool charX = new_char_list[i];
                float[] timestamp = timestamp_list[i];

                if (charX)
                {
                    timestamps.Add(new int[] {
                        (int)(timestamp[0] * 1000),
                        (int)(timestamp[1] * 1000)
                    });
                }
            }
#endif
            return timestamps;
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
                    string currText = _tokens[token].Split('\t')[0];
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
                    if (_offlineModel != null)
                    {
                        _offlineModel.Dispose();
                        _offlineModel = null;
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