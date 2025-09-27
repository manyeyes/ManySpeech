using FireRedAsr.Examples.Utils;

namespace FireRedAsr.Examples
{
    internal static partial class Program
    {
        public static FireRedAsr.OfflineRecognizer initFireRedAsrOfflineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            FireRedAsr.OfflineRecognizer offlineRecognizer = new FireRedAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, mvnFilePath, tokensFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void test_FireRedAsrOfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "fireredasr-aed-large-zh-en-onnx-offline-20250124";
            FireRedAsr.OfflineRecognizer offlineRecognizer = initFireRedAsrOfflineRecognizer(modelName);
            TimeSpan total_duration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int i = 0; i < 3; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    //method 1
                    //TimeSpan duration = TimeSpan.Zero;
                    //float[] sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
                    //samples.Add(sample);
                    //samplesList.Add(samples);
                    //total_duration += duration;
                    //method 2
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration);
                    samplesList.Add(samples);
                    total_duration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<FireRedAsr.OfflineStream> streams = new List<FireRedAsr.OfflineStream>();
            // method 1
            //foreach (var sample in samples)
            //{
            //    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
            //    stream.AddSamples(sample);
            //    streams.Add(stream);
            //}
            // method 2
            foreach (List<float[]> samplesListItem in samplesList)
            {
                FireRedAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            // decode,fit batch=1
            foreach (FireRedAsr.OfflineStream stream in streams)
            {
                FireRedAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                Console.WriteLine(result.Text);
                Console.WriteLine("");
            }
            //fit batch>1,but all in one
            //List<FireRedAsr.Model.OfflineRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            //foreach (FireRedAsr.Model.OfflineRecognizerResultEntity result in results_batch)
            //{
            //    Console.WriteLine(result.Text);
            //    Console.WriteLine("");
            //}
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
