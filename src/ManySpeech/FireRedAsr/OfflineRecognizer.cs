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

        public OfflineRecognizer(string encoderFilePath, string decoderFilePath, string mvnFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 1)
        {
            _mvnFilePath = mvnFilePath;
            AsrModel asrModel = new AsrModel(encoderFilePath, decoderFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
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
                for (int i = 0; i < maxlen; i++)
                {
                    DecoderOutputEntity decoderOutputEntity = _asrProj.DecoderProj(tokensList, encoder_outputs, src_mask, stackStatesList);
                    // 合并对应索引的子列表
                    tokensList = tokensList.Zip(decoderOutputEntity.TokensList, (a, b) =>a.Concat(b).ToList()).ToList();
                    stackStatesList = decoderOutputEntity.CacheList;
                    bool allEnd = tokensList.All(item => item.Any() && item.Last() == _asrProj.Eos_id);
                    if (allEnd)
                    {
                        break;
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
            List<OfflineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OfflineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OfflineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 4)
                    {
                        break;
                    }
                    string currToken = _tokens[token].Split(' ')[0];
                    if (currToken != "</s>" && currToken != "<s>" && currToken != "<sos/eos>" && currToken != "<blank>" && currToken != "<unk>" && currToken!= "<sos>" && currToken != "<eos>" && currToken != "<pad>")
                    {
                        if (IsChinese(currToken, true))
                        {
                            text_result += currToken;
                        }
                        else
                        {
                            text_result += "▁" + currToken + "▁";
                        }
                    }
                }
                OfflineRecognizerResultEntity onlineRecognizerResultEntity = new OfflineRecognizerResultEntity();
                onlineRecognizerResultEntity.Text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", "").Replace("▁", "").ToLower();
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