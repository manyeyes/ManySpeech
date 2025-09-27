// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using Microsoft.ML.OnnxRuntime.Tensors;
using ManySpeech.MoonshineAsr.Model;
using AliFsmnVad;


namespace ManySpeech.MoonshineAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    public class OnlineVadRecognizer : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private IAsrProj _asrProj;
        private FsmnVad? _fsmnVad = null;

        public OnlineVadRecognizer(string preprocessFilePath, string encodeFilePath, string cachedDecodeFilePath, string uncachedDecodeFilePath, string tokensFilePath, string vadModelFilePath, string vadConfigFilePath, string vadMvnFilePath, string configFilePath = "", int threadsNum = 1)
        {
            AsrModel asrModel = new AsrModel(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = Utils.PreloadHelper.ReadTokens(tokensFilePath);
            _asrProj = new AsrProjOfTransformer(asrModel);
            try
            {
                _fsmnVad = new FsmnVad(vadModelFilePath, vadConfigFilePath, vadMvnFilePath, 1);
            }
            catch (Exception ex)
            {
                //
            }
        }

        public OnlineVadStream CreateOnlineVadStream()
        {
            OnlineVadStream onlineVadStream = new OnlineVadStream(_asrProj, _fsmnVad);
            return onlineVadStream;
        }

        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineVadStream> streams)
        {
            this.Forward(streams);
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = this.DecodeMulti(streams);

            return onlineRecognizerResultEntities;
        }

        private void Forward(List<OnlineVadStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OnlineVadStream> streamsWorking = new List<OnlineVadStream>();
            int contextSize = 2;
            List<AsrInputEntity> modelInputs = new List<AsrInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            int padFrameNum = 0;
            int shiftFrameNum = 0;
            List<List<SegmentEntity>> segments = new List<List<SegmentEntity>>();
            List<List<int>> tokens = new List<List<int>>();
            List<List<int[]>> timestamps = new List<List<int[]>>();
            List<int[]> timesList = new List<int[]>();
            List<OnlineVadStream> streamsTemp = new List<OnlineVadStream>();
            List<int[]> nextTokens = new List<int[]>();
            List<int> seqLens = new List<int>();
            foreach (OnlineVadStream stream in streams)
            {
                AsrInputEntity asrInputEntity = new AsrInputEntity();
                int[] times = new int[2];
                if (stream.Segments.Count > 0)
                {
                    times = new int[2] { (int)stream.Segments.Last().Start, (int)stream.Segments.Last().End };
                }
                asrInputEntity.Speech = stream.GetDecodeChunk(ref padFrameNum, ref times);
                timesList.Add(times);
                shiftFrameNum = padFrameNum;
                if (asrInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                asrInputEntity.SpeechLength = asrInputEntity.Speech.Length;
                modelInputs.Add(asrInputEntity);
                stream.RemoveChunk(shiftFrameNum);
                statesList.Add(stream.States);
                segments.Add(stream.Segments);
                tokens.Add(new List<int>(stream.Tokens));
                timestamps.Add(new List<int[]>(stream.Timestamps));
                streamsWorking.Add(stream);
                nextTokens.Add(new int[] { 1 });
                seqLens.Add(1);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineVadStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                int batchSize = modelInputs.Count;
                int offset = streams[0].Offset;
                List<float[]> stackStatesList = new List<float[]>();
                stackStatesList = _asrProj.stack_states(statesList);
                PreprocessOutputEntity preprocessOutputEntity = _asrProj.PreprocessProj(modelInputs);
                EncodeOutputEntity encoderOutputEntity = _asrProj.EncodeProj(preprocessOutputEntity);
                int seqLen = 1;
                UncachedDecodeOutputEntity decoderOutputEntity = _asrProj.UncachedDecodeProj(encoderOutputEntity, nextTokens: nextTokens, seqLens: new List<int> { seqLen });
                Tensor<float>? logits_tensor = decoderOutputEntity.ReversibleEmbedding;
                List<float[]> cacheList = decoderOutputEntity.CacheList;
                List<int> indexList = new List<int>();
                for (int x = 0; x < (int)((double)modelInputs.MaxBy(x => x.Speech.Length).Speech.Length / 16000) * 6; x++)
                {
                    List<int[]> tokens_list = new List<int[]> { };
                    List<List<int[]>> timestamps_list = new List<List<int[]>>();
                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        int[] tokens_list_item = new int[logits_tensor.Dimensions[1]];
                        List<int[]> timestamps_list_item = new List<int[]>();
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            tokens_list_item[j] = (int)token_num;
                            timestamps_list_item.Add(new int[] { 0, 0 });
                        }
                        tokens_list.Add(tokens_list_item);
                        timestamps_list.Add(timestamps_list_item);
                    }
                    nextTokens = new List<int[]>();
                    for (int i = 0; i < tokens_list.Count; i++)
                    {
                        if (tokens_list[i].Last() == 2 || tokens_list[i].Last() == 0)
                        {
                            if (!indexList.Contains(i))
                            {
                                indexList.Add(i);
                            }
                        }
                        if (!indexList.Contains(i))
                        {
                            tokens[i].Add(tokens_list[i].Last());
                            timestamps[i].Add(timestamps_list[i].Last());
                            nextTokens.Add(new int[] { tokens_list[i].Last() });
                        }
                        else
                        {
                            if (tokens[i].Last() != 2)
                            {
                                tokens[i].Add(2);
                                timestamps[i].Add(new int[] { 0, 0 });
                                nextTokens.Add(new int[] { 0 });
                            }
                            else
                            {
                                tokens[i].Add(0);
                                timestamps[i].Add(new int[] { 0, 0 });
                                nextTokens.Add(new int[] { 0 });
                            }
                        }
                        seqLens[i] += 1;
                    }
                    if (indexList.Count == tokens_list.Count)
                    {
                        break;
                    }
                    seqLen = seqLen + 1;
                    CachedDecodeOutputEntity cachedDecodeOutputEntity = _asrProj.CachedDecodeProj(encoderOutputEntity, nextTokens: nextTokens, seqLens: new List<int> { seqLen }, cacheList: cacheList);
                    logits_tensor = cachedDecodeOutputEntity.ReversibleEmbedding;
                    cacheList = cachedDecodeOutputEntity.CacheList;
                }
                int streamIndex = 0;
                foreach (OnlineVadStream stream in streamsWorking)
                {
                    SegmentEntity segmentEntity = new SegmentEntity();
                    segmentEntity.Tokens = tokens[streamIndex].ToList();
                    segmentEntity.Timestamps = timestamps[streamIndex].ToList();
                    segmentEntity.Start = timesList[streamIndex][0];
                    segmentEntity.End = timesList[streamIndex][1];
                    stream.Segments.Add(segmentEntity);
                    stream.InSegmentsQueue(segmentEntity);
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }

        }
        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineVadStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。

            foreach (var stream in streams)
            {
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                string text_result = "";
                string lastToken = "";
                int[] lastTimestamp = null;
                if (stream.Segments.Count == 0)
                {
                    break;
                }
                SegmentEntity segmentEntity = stream.OutSegmentsQueue();
                if (segmentEntity == null)
                {
                    break;
                }
#if NET6_0_OR_GREATER || NETCOREAPP3_1_OR_GREATER
                foreach (var result in segmentEntity.Tokens.Zip<int, int[]>(segmentEntity.Timestamps))
                {
                    int token = result.First;
                    int[] timestamp = result.Second;
#else
                for (int i = 0; i < segmentEntity.Tokens.Count && i < segmentEntity.Timestamps.Count; i++)
                {
                    int token = segmentEntity.Tokens[i];
                    int[] timestamp = segmentEntity.Timestamps[i];
#endif
                    if (token == 2)
                    {
                        //break;
                    }
                    string currText = _tokens[token].Split('\t')[0];
                    if (currText != "</s>" && currText != "<s>" && currText != "<blank>" && currText != "<unk>")
                    {
                        if (Utils.ResultHelper.IsChinese(currText, true))
                        {
                            text_result += currText;
                            onlineRecognizerResultEntity.Tokens.Add(currText);
                            onlineRecognizerResultEntity.Timestamps.Add(timestamp);
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
                                onlineRecognizerResultEntity.Tokens.Remove(onlineRecognizerResultEntity.Tokens.Last());
                                onlineRecognizerResultEntity.Tokens.Add(currToken.Replace("▁", ""));
                                onlineRecognizerResultEntity.Timestamps.Remove(onlineRecognizerResultEntity.Timestamps.Last());
                                onlineRecognizerResultEntity.Timestamps.Add(currTimestamp);
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
                                if (onlineRecognizerResultEntity.Tokens.Count > 0)
                                {
                                    onlineRecognizerResultEntity.Tokens.Remove(onlineRecognizerResultEntity.Tokens.Last());
                                }
                                onlineRecognizerResultEntity.Tokens.Add(currToken.Replace("▁", ""));
                                if (onlineRecognizerResultEntity.Timestamps.Count > 0)
                                {
                                    onlineRecognizerResultEntity.Timestamps.Remove(onlineRecognizerResultEntity.Timestamps.Last());
                                }
                                onlineRecognizerResultEntity.Timestamps.Add(currTimestamp);
                                lastToken = currToken;
                                lastTimestamp = currTimestamp;
                            }
                            else
                            {
                                onlineRecognizerResultEntity.Tokens.Add(currText.Replace("▁", ""));
                                onlineRecognizerResultEntity.Timestamps.Add(timestamp);
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
                stream.Segments.Last().Text = Utils.ResultHelper.CheckText(text_result);
                onlineRecognizerResultEntity.Segments = stream.Segments;
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。
            return onlineRecognizerResultEntities;
        }

        public void DisposeOnlineVadStream(OnlineVadStream onlineVadStream)
        {
            if (onlineVadStream != null)
            {
                onlineVadStream.Dispose();
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
        ~OnlineVadRecognizer()
        {
            Dispose(_disposed);
        }
    }
}