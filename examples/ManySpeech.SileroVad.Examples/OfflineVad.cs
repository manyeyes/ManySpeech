using PreProcessUtils;

namespace ManySpeech.SileroVad.Examples
{
    internal static partial class Program
    {
        private static OfflineVad? _offlineVad = null;
        public static OfflineVad? initOfflineVad(string modelName)
        {
            try
            {
                if (_offlineVad == null)
                {
                    TimeSpan start_time0 = new TimeSpan(DateTime.Now.Ticks);
                    string modelFilePath = applicationBase + "./" + modelName + "/silero_vad.onnx";
                    string configFilePath = applicationBase + "./" + modelName + "/vad.yaml";
                    int batchSize = 2;
                    _offlineVad = new OfflineVad(modelFilePath, configFilePath: configFilePath, threshold: 0F, isDebug: false);
                    TimeSpan end_time0 = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds0 = end_time0.TotalMilliseconds - start_time0.TotalMilliseconds;
                    Console.WriteLine("load vad model elapsed_milliseconds:{0}", elapsed_milliseconds0.ToString());
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
            return _offlineVad;
        }
        public static void TestOfflineVad(List<float[]>? samples = null)
        {
            //string modelName = "silero-vad-onnx";
            //string modelName = "silero-vad-v5-onnx";
            string modelName = "silero-vad-v6-onnx";
            OfflineVad? offlineVad = initOfflineVad(modelName);
            if (offlineVad == null)
            {
                Console.WriteLine("Please check if the model is correct");
                return;
            }
            TimeSpan total_duration = new TimeSpan(0L);
            if (samples == null)
            {
                samples = new List<float[]>();
                int batchSize = 1;
                int startIndex = 2;
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/example/{0}.wav", n.ToString());//vad_example
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    TimeSpan duration = TimeSpan.Zero;
                    float[]? sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
                    samples.Add(sample);
                    total_duration += duration;
                }
            }

            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            // batch stream decode
            List<OfflineStream> streams = new List<OfflineStream>();
            foreach (float[] samplesItem in samples)
            {
                OfflineStream stream = offlineVad.CreateOfflineStream();
                stream.AddSamples(samplesItem);
                streams.Add(stream);
            }

            Console.WriteLine("vad infer result:");
            List<SileroVad.Model.VadResultEntity> results = offlineVad.GetResults(streams);
            foreach (SileroVad.Model.VadResultEntity result in results)
            {
                foreach (var item in result.Segments.Zip(result.Waveforms))
                {
                    Console.WriteLine(string.Format("{0}-->{1}", TimeSpan.FromMilliseconds(item.First.Start / 16).ToString(@"hh\:mm\:ss\,fff"), TimeSpan.FromMilliseconds(item.First.End / 16).ToString(@"hh\:mm\:ss\,fff")));
                    OfflineRecognizer(new List<float[]>() { item.Second });
                    Console.WriteLine("");
                }

            }
            
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("------------------------");
        }
    }
}
