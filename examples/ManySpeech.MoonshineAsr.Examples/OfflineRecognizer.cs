using ManySpeech.MoonshineAsr.Model;
using PreProcessUtils;

namespace ManySpeech.MoonshineAsr.Examples
{
    internal static partial class Program
    {
        public static OfflineRecognizer initMoonshineAsrOfflineRecognizer(string modelName)
        {
            string preprocessFilePath = applicationBase + "./" + modelName + "/preprocess.int8.onnx";
            string encodeFilePath = applicationBase + "./" + modelName + "/encode.int8.onnx";
            string cachedDecodeFilePath = applicationBase + "./" + modelName + "/cached_decode.int8.onnx";
            string uncachedDecodeFilePath = applicationBase + "./" + modelName + "/uncached_decode.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void test_MoonshineAsrOfflineRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "moonshine-tiny-en-onnx";
            string modelName = "moonshine-base-en-onnx";
            OfflineRecognizer offlineRecognizer = initMoonshineAsrOfflineRecognizer(modelName);
            TimeSpan total_duration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int i = 0; i < 5; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    //TimeSpan duration = TimeSpan.Zero;
                    //float[] sample = SpeechProcessing.AudioHelper.GetFileSample(wavFilePath, ref duration);
                    //samples.Add(sample);
                    //total_duration += duration;
                    // method 2
                    //TimeSpan duration = TimeSpan.Zero;
                    //samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration);
                    //samplesList.Add(samples);
                    //total_duration += duration;
                    // method 3
                    TimeSpan duration = TimeSpan.Zero;
                    float[] sample = AudioHelper.GetMediaSample(wavFilePath, ref duration);
                    samples = new List<float[]>();
                    samples.Add(sample);
                    samplesList.Add(samples);
                    total_duration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<MoonshineAsr.OfflineStream> streams = new List<MoonshineAsr.OfflineStream>();

            // fit method 1
            // assemble streams
            //foreach (var sample in samples)
            //{
            //    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
            //    stream.AddSamples(sample);
            //    streams.Add(stream);
            //}
            // fit method 2 or method 3
            foreach (List<float[]> samplesListItem in samplesList)
            {
                MoonshineAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            //fit batch> 1,but all in one
            List<OfflineRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            foreach (OfflineRecognizerResultEntity result in results_batch)
            {
                Console.WriteLine(result.Text);
                Console.WriteLine("");
            }
            // decode,fit batch=1
            foreach (OfflineStream stream in streams)
            {
                OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                Console.WriteLine(result.Text);
                Console.WriteLine("");
            }

            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("Hello, World!");
        }
        
    }
}
