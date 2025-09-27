using PreProcessUtils;
using ManySpeech.SileroVad.Model;

namespace ManySpeech.SileroVad.Examples
{
    internal static partial class Program
    {
        private static OnlineVad? _onlineVad = null;
        public static OnlineVad? initOnlineVad(string modelName)
        {
            try
            {
                if (_onlineVad == null)
                {
                    TimeSpan start_time0 = new TimeSpan(DateTime.Now.Ticks);
                    string modelFilePath = applicationBase + "./" + modelName + "/silero_vad.onnx";
                    string configFilePath = applicationBase + "./" + modelName + "/vad.yaml";
                    int batchSize = 2;
                    _onlineVad = new OnlineVad(modelFilePath, configFilePath: configFilePath, threshold: 0F, isDebug: false);
                    TimeSpan end_time0 = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds0 = end_time0.TotalMilliseconds - start_time0.TotalMilliseconds;
                    Console.WriteLine("load vad model elapsed_milliseconds:{0}", elapsed_milliseconds0.ToString());
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
            return _onlineVad;
        }
        public static void TestOnlineVad(List<float[]>? samples = null)
        {
            //string modelName = "silero-vad-onnx";
            //string modelName = "silero-vad-v5-onnx";
            string modelName = "silero-vad-v6-onnx";
            OnlineVad? onlineVad = initOnlineVad(modelName);
            if (onlineVad == null)
            {
                return;
            }
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;
            start_time = new TimeSpan(DateTime.Now.Ticks);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                samples = new List<float[]>();
                int batchSize = 1;
                int startIndex = 1;
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/example/{0}.wav", n.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize: 960);
                    for (int j = 0; j <1200; j++)
                    {
                        samples.Add(new float[960]);
                    }
                    samplesList.Add(samples);
                    total_duration += duration;
                    // method 2
                    //List<TimeSpan> durations = new List<TimeSpan>();
                    //samples = SpeechProcessing.AudioHelper.GetMediaChunkSamples(wavFilePath, ref durations);
                    //samplesList.Add(samples);
                    //foreach(TimeSpan duration in durations)
                    //{
                    //    total_duration += duration;
                    //}
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            // one stream decode
            //for (int j = 0; j < samplesList.Count; j++)
            //{
            //    OnlineVad.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
            //    foreach (float[] samplesItem in samplesList[j])
            //    {
            //        stream.AddSamples(samplesItem);
            //        OnlineVad.OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
            //        Console.WriteLine(result.text);
            //    }
            //    // 1
            //    //int w = 0;
            //    //while (w < 17)
            //    //{
            //    //    w++;
            //    //}
            //}

            //multi streams decode
            List<OnlineStream> onlineStreams = new List<OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                OnlineStream stream = onlineVad.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(true);
                isEnds.Add(false);
            }
            int i = 0;
            List<OnlineStream> streams = new List<OnlineStream>();

            while (true)
            {
                streams = new List<OnlineStream>();

                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (samplesList[j].Count > i && samplesList.Count > j)
                    {
                        onlineStreams[j].AddSamples(samplesList[j][i]);
                        streams.Add(onlineStreams[j]);
                        isEndpoints[j] = false;
                    }
                    else
                    {
                        streams.Add(onlineStreams[j]);
                        samplesList.Remove(samplesList[j]);
                        isEndpoints[j] = true;
                    }
                }
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (isEndpoints[j])
                    {
                        if (onlineStreams[j].IsFinished(isEndpoints[j]))
                        {
                            isEnds[j] = true;
                        }
                        else
                        {
                            streams.Add(onlineStreams[j]);
                        }
                    }
                }
                List<VadResultEntity> results_batch = onlineVad.GetResults(streams);
                foreach (VadResultEntity result in results_batch)
                {
                    //if (result.Waveforms.Count > waveformsNum)
                    //{
                    //    waveformsNum = result.Waveforms.Count;
                    //    test_AliParaformerAsrOfflineRecognizer(new List<float[]>() { result.Waveforms.Last() });
                    //    segments_samples.AddRange(result.Waveforms);
                    //}
                    //Console.WriteLine(result.text);
                    if (result.Waveforms.Count > 0)
                    {
                        if (result.Waveforms.Last() != null)
                        {
                            Console.WriteLine(string.Format("{0}-->{1}", TimeSpan.FromMilliseconds(result.Segments.Last().Start / 16).ToString(@"hh\:mm\:ss\,fff"), TimeSpan.FromMilliseconds(result.Segments.Last().End / 16).ToString(@"hh\:mm\:ss\,fff")));
                            OfflineRecognizer(new List<float[]>() { result.Waveforms.Last() });
                            Console.WriteLine("");
                        }
                        if (result == results_batch.Last())
                        {
                            Console.WriteLine("------------------------------");
                        }
                    }
                }
                //Console.WriteLine("");
                i++;
                bool isAllFinish = true;
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (!isEnds[j])
                    {
                        isAllFinish = false;
                        break;
                    }
                }
                if (isAllFinish)
                {
                    break;
                }
            }
            end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("------------------------");
        }
    }
}
