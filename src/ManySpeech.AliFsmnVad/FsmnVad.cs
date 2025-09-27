// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliFsmnVad.Model;
using ManySpeech.AliFsmnVad.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.AliFsmnVad
{
    /// <summary>
    /// FsmnVad
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class FsmnVad : IDisposable
    {
        // To detect redundant calls
        private bool _disposed;

        private InferenceSession _onnxSession;
        //private string _frontend;
        private WavFrontend _wavFrontend;
        private int _batchSize = 1;
        private int _max_end_sil = int.MinValue;
        private EncoderConfEntity _encoderConfEntity;
        private VadPostConfEntity _vad_post_conf;

        public FsmnVad(string modelFilePath, string configFilePath, string mvnFilePath, int threadsNum = 1, int batchSize = 1)
        {
            VadModel vadModel = new VadModel(modelFilePath, threadsNum: threadsNum);
            _onnxSession = vadModel.ModelSession;

            ConfEntity confEntity = LoadConf(configFilePath);// PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
            _wavFrontend = new WavFrontend(mvnFilePath, confEntity.frontend_conf);
            //_frontend = confEntity.frontend;
            _vad_post_conf = confEntity.vad_post_conf;
            _batchSize = batchSize;
            //_vad_scorer = new E2EVadModel(_vad_post_conf);
            _max_end_sil = _max_end_sil != int.MinValue ? _max_end_sil : confEntity.vad_post_conf.max_end_silence_time;
            _encoderConfEntity = confEntity.encoder_conf;
        }

        private ConfEntity? LoadConf(string configFilePath)
        {
            ConfEntity? confEntity = new ConfEntity();
            if (!string.IsNullOrEmpty(configFilePath))
            {
                if (configFilePath.ToLower().EndsWith(".json"))
                {
                    //confEntity = Utils.PreloadHelper.ReadJson<ConfEntity>(configFilePath);
                    confEntity = Utils.PreloadHelper.ReadJson(configFilePath);
                }
                else if (configFilePath.ToLower().EndsWith(".yaml"))
                {
                    confEntity = Utils.PreloadHelper.ReadYaml<ConfEntity>(configFilePath);
                }
            }
            return confEntity;
        }

        public SegmentEntity[] GetSegments(List<float[]> samples, int SpeechSpeed = 0)
        {
            if (SpeechSpeed > 0)
            {
                _max_end_sil = 160 * SpeechSpeed;
            }
            int waveform_nums = samples.Count;
            _batchSize = Math.Min(waveform_nums, _batchSize);
            SegmentEntity[] segments = new SegmentEntity[waveform_nums];
            for (int beg_idx = 0; beg_idx < waveform_nums; beg_idx += _batchSize)
            {
                int end_idx = Math.Min(waveform_nums, beg_idx + _batchSize);
                List<float[]> waveform_list = new List<float[]>();
                for (int i = beg_idx; i < end_idx; i++)
                {
                    waveform_list.Add(samples[i]);
                }
                List<VadInputEntity> vadInputEntitys = ExtractFeats(waveform_list);
                try
                {
                    int t_offset = 0;
                    int step = Math.Min(waveform_list.Max(x => x.Length), 6000);
                    bool is_final = true;
                    List<VadOutputEntity> vadOutputEntitys = Infer(vadInputEntitys);
                    for (int batch_num = beg_idx; batch_num < end_idx; batch_num++)
                    {
                        var scores = vadOutputEntitys[batch_num - beg_idx].Scores;
                        SegmentEntity[] segments_part = vadInputEntitys[batch_num].VadScorer.DefaultCall(scores, waveform_list[batch_num - beg_idx], is_final: is_final, max_end_sil: _max_end_sil, online: false);
                        if (segments_part.Length > 0)
                        {
#pragma warning disable CS8602 // 解引用可能出现空引用。
                            if (segments[batch_num] == null)
                            {
                                segments[batch_num] = new SegmentEntity();
                            }
                            if (segments_part[0] != null)
                            {
                                segments[batch_num].Segment.AddRange(segments_part[0].Segment); //
                            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

                        }
                    }
                }
                catch (OnnxRuntimeException ex)
                {
                    //_logger.LogWarning("input wav is silence or noise");
                    segments = null;
                }
                for (int batch_num = 0; batch_num < _batchSize; batch_num++)
                {
                    List<float[]> segment_waveforms = new List<float[]>();
                    foreach (int[] segment in segments[beg_idx + batch_num].Segment)
                    {
                        int frame_length = (((6000 * 400) / 400 - 1) * 160 + 400) / 60 / 1000;
                        int frame_start = segment[0] * frame_length;
                        int frame_end = segment[1] * frame_length;
                        float[] segment_waveform = new float[frame_end - frame_start];
                        Array.Copy(waveform_list[batch_num], frame_start, segment_waveform, 0, segment_waveform.Length);
                        segment_waveforms.Add(segment_waveform);
                    }
                    segments[beg_idx + batch_num].Waveform.AddRange(segment_waveforms);
                }
            }

            return segments;
        }

        public SegmentEntity[] GetSegmentsByStep(List<float[]> samples, int SpeechSpeed = 0)
        {
            if (SpeechSpeed > 0)
            {
                _max_end_sil = 160 * SpeechSpeed;
            }
            int waveform_nums = samples.Count;
            _batchSize = Math.Min(waveform_nums, _batchSize);
            SegmentEntity[] segments = new SegmentEntity[waveform_nums];
            for (int beg_idx = 0; beg_idx < waveform_nums; beg_idx += _batchSize)
            {
                int end_idx = Math.Min(waveform_nums, beg_idx + _batchSize);
                List<float[]> waveform_list = new List<float[]>();
                for (int i = beg_idx; i < end_idx; i++)
                {
                    float[] newSample = new float[samples[i].Length + 960000];
                    newSample = newSample.Select(x => x == 0 ? -23.025850929940457F : x).ToArray();
                    Array.Copy(samples[i], 0, newSample, 0, samples[i].Length);
                    waveform_list.Add(newSample);
                }
                List<VadInputEntity> vadInputEntitys = ExtractFeats(waveform_list);
                int feats_len = vadInputEntitys.Max(x => x.SpeechLength);
                List<float[]> in_cache = new List<float[]>();
                in_cache = PrepareCache(in_cache);
                try
                {
                    int step = Math.Min(vadInputEntitys.Max(x => x.SpeechLength), 6000 * 400);
                    bool is_final = true;
                    for (int t_offset = 0; t_offset < (int)(feats_len); t_offset += Math.Min(step, feats_len - t_offset))
                    {

                        if (t_offset + step >= feats_len - 1)
                        {
                            step = feats_len - t_offset;
                            is_final = true;
                        }
                        else
                        {
                            is_final = false;
                        }
                        List<VadInputEntity> vadInputEntitys_step = new List<VadInputEntity>();
                        foreach (VadInputEntity vadInputEntity in vadInputEntitys)
                        {
                            VadInputEntity vadInputEntity_step = new VadInputEntity();
                            float[]? feats = vadInputEntity.Speech;
                            int curr_step = Math.Min(feats.Length - t_offset, step);
                            if (curr_step <= 0)
                            {
                                vadInputEntity_step.Speech = new float[32000];
                                vadInputEntity_step.SpeechLength = 0;
                                vadInputEntity_step.InCaches = in_cache;
                                vadInputEntity_step.Waveform = new float[(((int)(32000) / 400 - 1) * 160 + 400)];//+ 0->400
                                vadInputEntitys_step.Add(vadInputEntity_step);
                                continue;
                            }
                            float[]? feats_step = new float[curr_step];
                            Array.Copy(feats, t_offset, feats_step, 0, feats_step.Length);
                            float[]? waveform = vadInputEntity.Waveform;
                            if (Math.Min(waveform.Length, ((int)(t_offset + step) / 400 - 1) * 160 + 0) - t_offset / 400 * 160 <= 0)
                            {
                                break;
                            }
                            float[]? waveform_step = new float[Math.Min(waveform.Length, ((int)(t_offset + step) / 400 - 1) * 160 + 480) - t_offset / 400 * 160];
                            Array.Copy(waveform, t_offset / 400 * 160, waveform_step, 0, waveform_step.Length);
                            vadInputEntity_step.Speech = feats_step;
                            vadInputEntity_step.SpeechLength = feats_step.Length;
                            vadInputEntity_step.InCaches = vadInputEntity.InCaches;
                            vadInputEntity_step.Waveform = waveform_step;
                            vadInputEntitys_step.Add(vadInputEntity_step);
                        }
                        if (vadInputEntitys_step.Count == 0)
                        {
                            break;
                        }
                        List<VadOutputEntity> vadOutputEntitys = Infer(vadInputEntitys_step);
                        for (int batch_num = 0; batch_num < _batchSize; batch_num++)
                        {
                            vadInputEntitys[batch_num].InCaches = vadOutputEntitys[batch_num].OutCaches;
                            var scores = vadOutputEntitys[batch_num].Scores;
                            SegmentEntity[] segments_part = vadInputEntitys[batch_num].VadScorer.DefaultCall(scores, vadInputEntitys_step[batch_num].Waveform, is_final: is_final, max_end_sil: _max_end_sil, online: false);
                            if (segments_part.Length > 0)
                            {

#pragma warning disable CS8602 // 解引用可能出现空引用。
                                if (segments[beg_idx + batch_num] == null)
                                {
                                    segments[beg_idx + batch_num] = new SegmentEntity();
                                }
                                if (segments_part[0] != null)
                                {
                                    segments[beg_idx + batch_num].Segment.AddRange(segments_part[0].Segment);
                                }
#pragma warning restore CS8602 // 解引用可能出现空引用。

                            }
                        }
                    }
                }
                catch (OnnxRuntimeException ex)
                {
                    //_logger.LogWarning("input wav is silence or noise");
                    segments = null;
                }
                catch (Exception ex)
                {
                    //_logger.LogError(ex.ToString());
                    segments = null;
                }
                if (segments != null)
                {
                    int frame_length = (((6000 * 400) / 400 - 1) * 160 + 400) / 60 / 1000;
                    for (int batch_num = 0; batch_num < _batchSize; batch_num++)
                    {
                        List<float[]> segment_waveforms = new List<float[]>();
                        foreach (int[] segment in segments[beg_idx + batch_num].Segment)
                        {
                            int frame_start = segment[0] * frame_length;
                            int frame_end = segment[1] * frame_length;
                            float[] segment_waveform = new float[frame_end - frame_start];
                            if (frame_end >= waveform_list[batch_num].Length - 960000)
                            {
                                frame_end = waveform_list[batch_num].Length - 960000;
                                if (frame_end > frame_start)
                                {
                                    segment_waveform = new float[frame_end - frame_start];
                                    Array.Copy(waveform_list[batch_num], frame_start, segment_waveform, 0, segment_waveform.Length);
                                    segment_waveforms.Add(segment_waveform);
                                }
                                break;
                            }
                            Array.Copy(waveform_list[batch_num], frame_start, segment_waveform, 0, segment_waveform.Length);
                            segment_waveforms.Add(segment_waveform);
                        }
                        segments[beg_idx + batch_num].Waveform.AddRange(segment_waveforms);
                    }
                }
                waveform_list = new List<float[]>();
                waveform_list = null;
                vadInputEntitys = null;
                in_cache = null;
            }

            return segments;
        }

        private List<float[]> PrepareCache(List<float[]> in_cache)
        {
            if (in_cache.Count > 0)
            {
                return in_cache;
            }

            int fsmn_layers = _encoderConfEntity.fsmn_layers;

            int proj_dim = _encoderConfEntity.proj_dim;
            int lorder = _encoderConfEntity.lorder;

            for (int i = 0; i < fsmn_layers; i++)
            {
                float[] cache = new float[1 * proj_dim * (lorder - 1) * 1];
                in_cache.Add(cache);
            }
            return in_cache;
        }

        private List<VadInputEntity> ExtractFeats(List<float[]> waveform_list)
        {
            List<float[]> in_cache = new List<float[]>();
            in_cache = PrepareCache(in_cache);
            List<VadInputEntity> vadInputEntitys = new List<VadInputEntity>();
            foreach (var waveform in waveform_list)
            {
                float[] fbanks = _wavFrontend.GetFbank(waveform);
                float[] features = _wavFrontend.LfrCmvn(fbanks);
                VadInputEntity vadInputEntity = new VadInputEntity();
                vadInputEntity.Waveform = waveform;
                vadInputEntity.Speech = features;
                vadInputEntity.SpeechLength = features.Length;
                vadInputEntity.InCaches = in_cache;
                vadInputEntity.VadScorer = new E2EVadModel(_vad_post_conf);
                vadInputEntitys.Add(vadInputEntity);
            }
            return vadInputEntitys;
        }
        /// <summary>
        /// 一维数组转3维数组
        /// </summary>
        /// <param name="obj"></param>
        /// <param name="len">一维长</param>
        /// <param name="wid">二维长</param>
        /// <returns></returns>
        public static T[,,] DimOneToThree<T>(T[] oneDimObj, int len, int wid)
        {
            if (oneDimObj.Length % (len * wid) != 0)
                return null;
            int height = oneDimObj.Length / (len * wid);
            T[,,] threeDimObj = new T[len, wid, height];

            for (int i = 0; i < oneDimObj.Length; i++)
            {
                //核心思想把握每个维度的值多久变一次与设置最大值，变化频率设置用除法，设置最大值用求余
                //第二及之后的维度最大值为自身维度最大值 -1（意思就是最后需要对自身维度最大值求余）
                threeDimObj[i / (wid * height), (i / height) % wid, i % height] = oneDimObj[i];
            }
            return threeDimObj;
        }

        private List<VadOutputEntity> Infer(List<VadInputEntity> vadInputEntitys)
        {
            List<VadOutputEntity> vadOutputEntities = new List<VadOutputEntity>();
            try
            {
                foreach (VadInputEntity vadInputEntity in vadInputEntitys)
                {
                    int batchSize = 1;//_batchSize                
                    var inputMeta = _onnxSession.InputMetadata;
                    var container = new List<NamedOnnxValue>();
                    int[] dim = new int[] { batchSize, vadInputEntity.Speech.Length / 400 / batchSize, 400 };
                    var tensor = new DenseTensor<float>(vadInputEntity.Speech, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>("speech", tensor));

                    int i = 0;
                    foreach (var cache in vadInputEntity.InCaches)
                    {
                        int[] cache_dim = new int[] { 1, 128, cache.Length / 128 / 1, 1 };
                        var cache_tensor = new DenseTensor<float>(cache, cache_dim, false);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>("in_cache" + i.ToString(), cache_tensor));
                        i++;
                    }

                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
                    var resultsArray = results.ToArray();
                    VadOutputEntity vadOutputEntity = new VadOutputEntity();
                    for (int j = 0; j < resultsArray.Length; j++)
                    {
                        if (resultsArray[j].Name.Equals("logits"))
                        {
                            Tensor<float> tensors = resultsArray[0].AsTensor<float>();
                            var _scores = DimOneToThree<float>(tensors.ToArray(), 1, tensors.Dimensions[1]);
                            vadOutputEntity.Scores = _scores;
                        }
                        if (resultsArray[j].Name.StartsWith("out_cache"))
                        {
                            vadOutputEntity.OutCaches.Add(resultsArray[j].AsEnumerable<float>().ToArray());
                        }

                    }
                    vadOutputEntities.Add(vadOutputEntity);
                }
            }
            catch (Exception ex)
            {
                //
            }

            return vadOutputEntities;
        }
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    if (_onnxSession != null)
                    {
                        _onnxSession.Dispose();
                    }
                    if (_wavFrontend != null)
                    {
                        _wavFrontend.Dispose();
                    }
                    if (_encoderConfEntity != null)
                    {
                        _encoderConfEntity = null;
                    }
                    if (_vad_post_conf != null)
                    {
                        _vad_post_conf = null;
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
        ~FsmnVad()
        {
            Dispose(_disposed);
        }
    }
}