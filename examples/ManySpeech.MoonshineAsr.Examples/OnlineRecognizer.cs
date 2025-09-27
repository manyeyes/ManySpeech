using ManySpeech.MoonshineAsr.Model;
using PreProcessUtils;

namespace ManySpeech.MoonshineAsr.Examples
{
    internal static partial class Program
    {        

        public static OnlineRecognizer initMoonshineAsrOnlineRecognizer(string modelName)
        {
            string preprocessFilePath = applicationBase + "./" + modelName + "/preprocess.int8.onnx";
            string encodeFilePath = applicationBase + "./" + modelName + "/encode.int8.onnx";
            string cachedDecodeFilePath = applicationBase + "./" + modelName + "/cached_decode.int8.onnx";
            string uncachedDecodeFilePath = applicationBase + "./" + modelName + "/uncached_decode.int8.onnx";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OnlineRecognizer onlineRecognizer = new MoonshineAsr.OnlineRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, threadsNum: 1);
            return onlineRecognizer;
        }

        public static void test_MoonshineAsrOnlineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "moonshine-tiny-en-onnx";
            //string modelName = "moonshine-base-en-onnx";
            MoonshineAsr.OnlineRecognizer onlineRecognizer = initMoonshineAsrOnlineRecognizer(modelName);
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;


            List<List<float[]>> samplesList = new List<List<float[]>>();
            int batchSize = 1;
            int startIndex = 5;
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", n.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize: 160 * 6);
                    for (int j = 0; j < 30; j++)
                    {
                        samples.Add(new float[400]);
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
            start_time = new TimeSpan(DateTime.Now.Ticks);            
            List<MoonshineAsr.OnlineStream> onlineStreams = new List<MoonshineAsr.OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                MoonshineAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
            }
            int i = 0;
            List<MoonshineAsr.OnlineStream> streams = new List<MoonshineAsr.OnlineStream>();
            while (true)
            {
                streams = new List<MoonshineAsr.OnlineStream>();

                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (samplesList[j].Count > i && samplesList.Count > j)
                    {
                        onlineStreams[j].AddSamples(samplesList[j][i]);
                        streams.Add(onlineStreams[j]);
                        isEndpoints[0] = false;
                    }
                    else
                    {
                        streams.Add(onlineStreams[j]);
                        samplesList.Remove(samplesList[j]);
                        isEndpoints[0] = true;
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
                List<OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                foreach (OnlineRecognizerResultEntity result in results_batch)
                {
                    Console.WriteLine(string.Join("", result.Segments.SelectMany(x => x.Text).ToArray()));
                }
                Console.WriteLine("");
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
            Console.WriteLine("Hello, World!");
        }
    }
}
