using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad
{
    delegate void ForwardBatchOffline(List<OfflineStream> streams);
    public class OfflineVad : IDisposable
    {
        private bool _disposed;

        private IVadProj? _vadProj;
        private ForwardBatchOffline? _forwardBatch;

        /// <summary>
        /// OfflineVad
        /// </summary>
        /// <param name="modelFilePath">模型文件</param>
        /// <param name="configFilePath">配置文件</param>
        /// <param name="threshold">vad 阈值</param>
        /// <param name="sampleRate">采样率</param>
        /// <param name="threadsNum">onnx runtime 线程数</param>
        /// <param name="isDebug">是否输出调试日志</param>
        public OfflineVad(string modelFilePath, string configFilePath = "", float threshold = 0F, int sampleRate = 16000, int threadsNum = 2, bool isDebug = false)
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
            _forwardBatch = new ForwardBatchOffline(this.ForwardBatch);
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream offlineStream = new OfflineStream(_vadProj);
            return offlineStream;
        }
        public VadResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            VadResultEntity vadResultEntity = GetResults(streams)[0];

            return vadResultEntity;
        }

        public List<VadResultEntity> GetResults(List<OfflineStream> streams)
        {
            List<VadResultEntity> segmentEntities = new List<VadResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            _forwardBatch.Invoke(streams);
#pragma warning restore CS8602 // 解引用可能出现空引用。
            segmentEntities = this.DecodeMulti(streams);
            return segmentEntities;
        }
        public void ForwardBatch(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            try
            {
                for (int i = 0; i < streams.Select(x => x.Waveform.Length).Max() / 512; i++)
                {
                    List<ModelInputEntity> modelInputs = new List<ModelInputEntity>();
                    List<List<float[]>> stateList = new List<List<float[]>>();
                    List<OfflineStream> streamsTemp = new List<OfflineStream>();
                    foreach (OfflineStream stream in streams)
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
                        continue;
                    }
                    foreach (OfflineStream stream in streamsTemp)
                    {
                        streams.Remove(stream);
                    }
                    List<float[]> states = new List<float[]>();
                    List<float[]> stackStatesList = new List<float[]>();
                    stackStatesList = _vadProj.stack_states(stateList);
                    ModelOutputEntity modelOutput = _vadProj.ModelProj(modelInputs, stackStatesList);

                    List<List<float[]>> next_statesList = new List<List<float[]>>();
                    next_statesList = _vadProj.unstack_states(modelOutput.ModelOutStates);
                    int streamIndex = 0;
                    foreach (OfflineStream stream in streams)
                    {
                        stream.PushIndex(modelOutput.ModelOut[streamIndex]);
                        stream.States = next_statesList[streamIndex];
                        streamIndex++;
                    }
                }
                foreach (OfflineStream stream in streams)
                {
                    stream.CalculateProb();
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Offline vad failed", ex);
            }
        }

        public List<VadResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<VadResultEntity> vadResultEntities = new List<VadResultEntity>();
            foreach (OfflineStream stream in streams)
            {
                VadResultEntity vadResultEntity = new VadResultEntity();
                vadResultEntity.Segments = stream.Segments;
                vadResultEntity.Waveforms = new List<float[]>();
                int i = 0;
                foreach (SegmentEntity segment in stream.Segments)
                {
                    i++;
                    int multiple = 1;
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
                }
                vadResultEntities.Add(vadResultEntity);
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
        ~OfflineVad()
        {
            Dispose(_disposed);
        }
    }
}
