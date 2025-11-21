using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad
{
    delegate void ForwardBatchOnline(List<OnlineStream> streams);
    public class OnlineVad : IDisposable
    {
        bool _disposed;

        private IVadProj? _vadProj;
        private ForwardBatchOnline? _forwardBatch;

        /// <summary>
        /// OnlineVad
        /// </summary>
        /// <param name="modelFilePath">模型文件</param>
        /// <param name="configFilePath">配置文件</param>
        /// <param name="threshold">vad 阈值</param>
        /// <param name="sampleRate">采样率</param>
        /// <param name="threadsNum">onnx runtime 线程数</param>
        /// <param name="isDebug">是否输出调试日志</param>
        public OnlineVad(string modelFilePath, string configFilePath = "", float threshold = 0F, int sampleRate = 16000, int threadsNum = 2, bool isDebug = false)
        {
            VadModel vadModel = new VadModel(modelFilePath, configFilePath: configFilePath, threshold: threshold, threadsNum: threadsNum);
            switch (vadModel.CustomMetadata.version)
            {
                case "v4":
                    _vadProj = new VadProj(vadModel, sampleRate, isDebug);
                    break;
                case "v5":
                case "v6":
                    _vadProj = new VadProjOfV5(vadModel, sampleRate, isDebug);
                    break;
            }
            _forwardBatch = new ForwardBatchOnline(this.ForwardBatch);
        }
        public OnlineStream CreateOnlineStream()
        {
            OnlineStream onlineStream = new OnlineStream(_vadProj);
            return onlineStream;
        }
        public VadResultEntity GetResult(OnlineStream stream)
        {
            List<OnlineStream> streams = new List<OnlineStream>();
            streams.Add(stream);
            VadResultEntity vadResultEntity = GetResults(streams).FirstOrDefault() ?? new VadResultEntity();

            return vadResultEntity;
        }

        public List<VadResultEntity> GetResults(List<OnlineStream> streams)
        {
            List<VadResultEntity> segmentEntities = new List<VadResultEntity>();
            _forwardBatch?.Invoke(streams);
            segmentEntities = this.DecodeMulti(streams);
            return segmentEntities;
        }
        public void ForwardBatch(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<ModelInputEntity> modelInputs = new List<ModelInputEntity>();
            List<List<float[]>> stateList = new List<List<float[]>>();
            List<OnlineStream> streamsTemp = new List<OnlineStream>();
            foreach (OnlineStream stream in streams)
            {
                ModelInputEntity modelInputEntity = new ModelInputEntity();
                modelInputEntity.Speech = stream.GetDecodeChunk(stream.ChunkLength);
                if (modelInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                modelInputEntity.SpeechLength = modelInputEntity.Speech.Length;
                modelInputs.Add(modelInputEntity);
                stateList.Add(stream.States);
                stream.RemoveChunk(stream.ShiftLength);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                List<float[]> states = new List<float[]>();
                List<float[]> stackStatesList = new List<float[]>();
                stackStatesList = _vadProj.stack_states(stateList);
                ModelOutputEntity modelOutput = _vadProj.ModelProj(modelInputs, stackStatesList);
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                next_statesList = _vadProj.unstack_states(modelOutput.ModelOutStates);
                int streamIndex = 0;
                foreach (OnlineStream stream in streams)
                {
                    stream.PushIndex(modelOutput.ModelOut[streamIndex]);
                    stream.States = next_statesList[streamIndex];
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Online vad failed", ex);
            }
        }
        public List<VadResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<VadResultEntity> vadResultEntities = new List<VadResultEntity>();
            foreach (OnlineStream stream in streams)
            {
                VadResultEntity vadResultEntity = new VadResultEntity();
                //vadResultEntity.Segments = stream.Segments;
                vadResultEntity.Segments = new List<SegmentEntity>();
                vadResultEntity.Waveforms = new List<float[]>();
                if (stream.Segments.Count > stream.LastWaveFormsNum)
                {
                    stream.LastWaveFormsNum = stream.Segments.Count;
                    int multiple = 1;
                    if (stream.Segments.Count > 0)
                    {
                        SegmentEntity segment = stream.Segments.Last();
                        int startExtendLen = _vadProj.CustomMetadata.segment_start_extend_len;
                        int endExtendLen = _vadProj.CustomMetadata.segment_end_extend_len;
                        int startExtend = segment.Start * multiple;
                        if (segment.Start * multiple - startExtendLen > 0)
                        {
                            startExtend = segment.Start * multiple - startExtendLen;
                        }
                        int endExtend = segment.End * multiple;
                        if (stream.Waveform.Length - startExtend - (segment.End * multiple - startExtend) > endExtendLen)
                        {
                            endExtend = segment.End * multiple + endExtendLen;
                        }
                        float[] waveformItem = new float[endExtend - startExtend];

                        Array.Copy(stream.Waveform, startExtend, waveformItem, segment.Start * multiple - startExtend, endExtend - segment.Start * multiple);
                        Array.Resize(ref waveformItem, waveformItem.Length + 2400);
                        Array.Copy(new float[400], 0, waveformItem, 0, 400);
                        vadResultEntity.Waveforms.Add(waveformItem);
                        vadResultEntity.Segments.Add(stream.Segments.LastOrDefault());
                        vadResultEntities.Add(vadResultEntity);
                    }
                }
            }
            return vadResultEntities;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_vadProj != null)
                    {
                        _vadProj.Dispose();
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
        ~OnlineVad()
        {
            Dispose(_disposed);
        }
    }
}
