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
            //string modelName = "whisper-tiny-onnx";
            //string modelName = "whisper-tiny-en-onnx";
            //string modelName = "whisper-base-onnx";
            //string modelName = "whisper-base-en-onnx";
            string modelName = "whisper-small-onnx";
            //string modelName = "whisper-small-en-onnx";
            //string modelName = "whisper-small-cantonese-cer10.1-onnx";
            //string modelName = "whisper-small-cantonese-cer7.93-onnx";
            //string modelName = "whisper-medium-onnx";
            //string modelName = "whisper-medium-en-onnx";
            //string modelName = "whisper-large-v1-onnx";
            //string modelName = "whisper-large-v2-onnx";
            //string modelName = "whisper-large-v2-multi-hans-onnx";
            //string modelName = "whisper-large-v3-onnx";
            //string modelName = "whisper-large-v3-zh-onnx-belle-20240311";
            //string modelName = "whisper-large-v3-turbo-onnx";
            //string modelName = "whisper-large-v3-turbo-zh-onnx-belle-20241016";
            //string modelName = "whisper-large-v3-turbo-ja-onnx-anime-20241110";
            //string modelName = "distil-whisper-small-en-onnx";
            //string modelName = "distil-whisper-medium-en-onnx";
            //string modelName = "distil-whisper-large-v2-en-onnx";
            //string modelName = "distil-whisper-large-v3-en-onnx";
            //string modelName = "distil-whipser-large-v3-en-onnx";
            //string modelName = "distil-whisper-large-v2-multi-hans-onnx";
            //string modelName = "distil-whisper-small-cantonese-onnx-alvanlii-20240404";
            //string modelName = "whisper-small-cantonese-onnx-alvanlii-20240515";
            //string modelName = "distil-whisper-large-v3-de-onnx-primeline-20240531";
            //string modelName = "medusa-whisper-large-v2-onnx";
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
                    // method 1
                    TimeSpan duration = TimeSpan.Zero;
                    //samples = AudioHelper.GetMediaChunkSamples(wavFilePath, ref duration, chunkSize: 160 * 6);
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize: 160 * 6);
                    for (int j = 0; j < 500; j++)
                    {
                        samples.Add(new float[960]);
                    }
                    samplesList.Add(samples);
                    totalDuration += duration;
                    // method 2
                    //List<TimeSpan> durations = new List<TimeSpan>();
                    ////samples = AudioHelper.GetMediaChunkSamples(wavFilePath, ref durations);
                    //samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref durations);
                    //samplesList.Add(samples);
                    //foreach(TimeSpan duration in durations)
                    //{
                    //    totalDuration += duration;
                    //}
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            start_time = new TimeSpan(DateTime.Now.Ticks);
            // one stream decode
            //for (int j = 0; j < samplesList.Count; j++)
            //{
            //    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
            //    foreach (float[] samplesItem in samplesList[j])
            //    {
            //        stream.AddSamples(samplesItem);
            //    }
            //    // 1
            //    int w = 0;
            //    while (w < 17)
            //    {
            //        OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
            //        Console.WriteLine(result_on.text);
            //        w++;
            //    }
            //    // 2
            //    //OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
            //    //Console.WriteLine(result_on.text);
            //}

            //multi streams decode
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
