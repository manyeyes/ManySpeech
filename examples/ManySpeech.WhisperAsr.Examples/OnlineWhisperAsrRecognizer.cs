using ManySpeech.WhisperAsr.Model;
using PreProcessUtils;
using System.Diagnostics;

namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        public static OnlineRecognizer initWhisperAsrOnlineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, threadsNum: 5);
            return onlineRecognizer;
        }

        public static void test_WhisperAsrOnlineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "whisper-tiny-onnx";
            OnlineRecognizer onlineRecognizer = initWhisperAsrOnlineRecognizer(modelName);
            TimeSpan totalDuration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;


            List<List<float[]>> samplesList = new List<List<float[]>>();
            int batchSize = 1;
            int startIndex = 2;
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./test_wavs/{0}.wav", n.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize: 160 * 6);
                    for (int j = 0; j < 500; j++)
                    {
                        samples.Add(new float[960]);
                    }
                    samplesList.Add(samples);
                    totalDuration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            start_time = new TimeSpan(DateTime.Now.Ticks);
            List<OnlineStream> onlineStreams = new List<OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            List<int> lastNums = new List<int>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
                lastNums.Add(0);
            }
            int i = 0;
            List<OnlineStream> streams = new List<OnlineStream>();

            Task t0 = new Task(() =>
            {
                while (true)
                {
                    List<string> texts = onlineRecognizer.GetStreamingTexts();
                    foreach (var text in texts)
                    {
                        Console.WriteLine(text);
                        Debug.WriteLine(text);
                    }
                    Task.Delay(500);
                }
            });
            t0.Start();

            while (true)
            {
                streams = new List<OnlineStream>();

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
                        onlineStreams[j].InputFinished = true;
                        onlineStreams[j].AddSamples(new float[0]);
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
                int n = 0;
                foreach (OnlineRecognizerResultEntity result in results_batch)
                {
                    if (lastNums[n] == result.Segments.Count)
                    {
                        n++;
                        continue;

                    }
                    foreach (var segment in result.Segments)
                    {
                        float? start = segment.Start;
                        float? end = segment.End;
                        string? text = segment.Text;
                        string line = string.Format("[{0}-->{1}] {2}", TimeSpan.FromSeconds((double)start).ToString(@"hh\:mm\:ss\,fff"), TimeSpan.FromSeconds((double)end).ToString(@"hh\:mm\:ss\,fff"), text);
                        Console.WriteLine(line);
                        System.Diagnostics.Debug.WriteLine(line);
                    }
                    Console.WriteLine(result.Text);
                    Console.WriteLine("");
                    lastNums[n] = result.Segments.Count;
                    n++;

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
            Console.WriteLine("=================================================");
            end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
            Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
            Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
        }
    }
}
