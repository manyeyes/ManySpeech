using ManySpeech.MoonshineAsr.Model;
using PreProcessUtils;

namespace ManySpeech.MoonshineAsr.Examples
{
    internal static partial class Program
    {
        public static OnlineVadRecognizer initMoonshineAsrOnlineVadRecognizer(string modelName,string vadModelName)
        {
            string preprocessFilePath = applicationBase + "./" + modelName + "/preprocess.onnx";
            string encodeFilePath = applicationBase + "./" + modelName + "/encode.onnx";
            string cachedDecodeFilePath = applicationBase + "./" + modelName + "/cached_decode.onnx";
            string uncachedDecodeFilePath = applicationBase + "./" + modelName + "/uncached_decode.onnx";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            string vadModelFilePath = applicationBase + "/" + vadModelName + "/" + "model.int8.onnx";
            string vadMvnFilePath = applicationBase + vadModelName + "/" + "vad.mvn";
            string vadConfigFilePath = applicationBase + vadModelName + "/" + "vad.json";
            OnlineVadRecognizer onlineVadRecognizer = new OnlineVadRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, vadModelFilePath, vadConfigFilePath, vadMvnFilePath, threadsNum: 1);
            return onlineVadRecognizer;
        }

        public static void test_MoonshineAsrOnlineVadRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "moonshine-tiny-en-onnx";
            string modelName = "moonshine-base-en-onnx";
            string vadModelName = "speech_fsmn_vad_zh-cn-16k-common-onnx";
            OnlineVadRecognizer onlineVadRecognizer = initMoonshineAsrOnlineVadRecognizer(modelName,vadModelName);
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
                    for (int j = 0; j < 400; j++)
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
            List<MoonshineAsr.OnlineVadStream> onlineStreams = new List<MoonshineAsr.OnlineVadStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                MoonshineAsr.OnlineVadStream stream = onlineVadRecognizer.CreateOnlineVadStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
            }
            int i = 0;
            List<MoonshineAsr.OnlineVadStream> streams = new List<MoonshineAsr.OnlineVadStream>();

            while (true)
            {
                streams = new List<MoonshineAsr.OnlineVadStream>();

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
                List<OnlineRecognizerResultEntity> results_batch = onlineVadRecognizer.GetResults(streams);
                foreach (OnlineRecognizerResultEntity result in results_batch)
                {
                    var seg = result.Segments.Last();
                    if (seg.Text.Length > 0)
                    {
                        string line = string.Format("[{0}-->{1}] \r\n {2}", TimeSpan.FromMilliseconds((double)seg.Start).ToString(@"hh\:mm\:ss\,fff"), TimeSpan.FromMilliseconds((double)seg.End).ToString(@"hh\:mm\:ss\,fff"), seg.Text);
                        Console.WriteLine(line);
                        Console.WriteLine("");
                    }
                }
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
