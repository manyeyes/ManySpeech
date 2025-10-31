// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.AudioSep.Model;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2025 by manyeyes
    /// </summary>
    public class OfflineSep : IDisposable
    {
        private bool _disposed;

        private string[] _tokens;
        private string _mvnFilePath;
        private ISepProj _sepProj;

        public OfflineSep(string modelFilePath,string generatorFilePath="", string configFilePath = "", int threadsNum = 1)
        {
            SepModel sepModel = new SepModel(modelFilePath:modelFilePath, generatorFilePath: generatorFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            switch (sepModel.ConfEntity?.model.model_type.ToLower())
            {
                case "gtcrnse":
                    _sepProj = new SepProjOfGtcrnSe(sepModel);
                    break;
                case "spleeter":
                    _sepProj = new SepProjOfSpleeter(sepModel);
                    break;
                case "uvr":
                    _sepProj = new SepProjOfUvr(sepModel);
                    break;
            }
        }

        public OfflineStream CreateOfflineStream()
        {
            OfflineStream onlineStream = new OfflineStream(_mvnFilePath, _sepProj);
            return onlineStream;
        }
        public OfflineSepResultEntity GetResult(OfflineStream stream)
        {
            List<OfflineStream> streams = new List<OfflineStream>();
            streams.Add(stream);
            OfflineSepResultEntity offlineSepResultEntity = GetResults(streams)[0];

            return offlineSepResultEntity;
        }
        public List<OfflineSepResultEntity> GetResults(List<OfflineStream> streams)
        {
            this.Forward(streams);
            List<OfflineSepResultEntity> offlineRecognizerResultEntities = this.DecodeMulti(streams);
            return offlineRecognizerResultEntities;
        }

        private void Forward(List<OfflineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OfflineStream> streamsWorking = new List<OfflineStream>();
            List<ModelInputEntity> modelInputs = new List<ModelInputEntity>();
            List<OfflineStream> streamsTemp = new List<OfflineStream>();
            foreach (OfflineStream stream in streams)
            {
                ModelInputEntity modelInputEntity = new ModelInputEntity();
                modelInputEntity.SampleRate = stream.SampleRate;
                modelInputEntity.Channels = stream.Channels;
                modelInputEntity.Speech = stream.GetDecodeChunk();
                if (modelInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                modelInputEntity.SpeechLength = modelInputEntity.Speech.Length;
                modelInputs.Add(modelInputEntity);
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
                List<ModelOutputEntity> modelOutputEntity = _sepProj.ModelProj(modelInputs);
                List<ModelOutputEntity> generatorOutputEntity = _sepProj.GeneratorProj(modelOutputEntity[0]);
                if (generatorOutputEntity != null)
                {
                    modelOutputEntity = generatorOutputEntity;
                }
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                int streamIndex = 0;
                foreach (OfflineStream stream in streamsWorking)
                {
                    if (stream.ModelOutputEntities == null)
                    {
                        stream.ModelOutputEntities = new List<ModelOutputEntity>();
                    }
                    stream.ModelOutputEntities.AddRange(modelOutputEntity);
                    stream.RemoveDecodedChunk();
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }

        }

        public static float[] FlattenList(List<float[]> list)
        {
            int totalElements = list.Sum(array => array.Length);
            float[] result = new float[totalElements];

            int offset = 0;
            foreach (float[] array in list)
            {
                array.CopyTo(result, offset);
                offset += array.Length;
            }

            return result;
        }

        private List<OfflineSepResultEntity> DecodeMulti(List<OfflineStream> streams)
        {
            List<OfflineSepResultEntity> offlineRecognizerResultEntities = new List<OfflineSepResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            try
            {
                foreach (var stream in streams)
                {
                    OfflineSepResultEntity offlineSepResultEntity = new OfflineSepResultEntity();
                    //string text_result = "";
                    string lastToken = "";
                    int[] lastTimestamp = null;
                    foreach (var stemName in stream.ModelOutputEntities.Select(x=>x.StemName).Distinct())
                    {
                        if (!offlineSepResultEntity.Stems.ContainsKey(stemName))
                        {
                            List<float[]?> sampleList = stream.ModelOutputEntities.Where(x => x.StemName == stemName).Select(x => x.StemContents).ToList();
                            offlineSepResultEntity.Stems.Add(stemName, FlattenList(sampleList));
                        }

                        //offlineSepResultEntity.Stems.Add(result.StemName, result.StemContents);
                    }
                    offlineSepResultEntity.AudioId = stream.AudioId;
                    offlineSepResultEntity.Channels = stream.Channels;
                    offlineSepResultEntity.SampleRate = stream.SampleRate;

                    offlineRecognizerResultEntities.Add(offlineSepResultEntity);
                }
            }
            catch (Exception ex)
            {

            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return offlineRecognizerResultEntities;
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
                    if (_sepProj != null)
                    {
                        _sepProj.Dispose();
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
        ~OfflineSep()
        {
            Dispose(_disposed);
        }
    }
}