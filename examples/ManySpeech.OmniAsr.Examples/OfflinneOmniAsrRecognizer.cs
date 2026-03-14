using ManySpeech.OmniAsr.Model;
using ManySpeech.OmniAsr;
using PreProcessUtils;

namespace ManySpeech.OmniAsr.Examples
{
    internal partial class OfflinneOmniAsrRecognizer : BaseAsr
    {
        public static OfflineRecognizer initOfflineRecognizer(string modelName)
        {
            string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
            //string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            string configFilePath = "";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, tokensFilePath: tokensFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void OfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "OmniASR-CTC-300M-int8-onnx";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 10; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "./test_wavs/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    TimeSpan duration = TimeSpan.Zero;
                    float[] sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
                    samples = new List<float[]>();
                    samples.Add(sample);
                    samplesList.Add(samples);
                    totalDuration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            OfflineRecognizer offlineRecognizer = initOfflineRecognizer(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<OfflineStream> streams = new List<OfflineStream>();
            foreach (List<float[]> samplesListItem in samplesList)
            {
                OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            List<OfflineRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            foreach (OfflineRecognizerResultEntity result in results_batch)
            {
                Console.WriteLine(result.Text);
                System.Diagnostics.Debug.WriteLine(result.Text);
                Console.WriteLine("");
                System.Diagnostics.Debug.WriteLine("");
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
            Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
            Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
        }
    }
}
