using ManySpeech.AudioTagging.Model;
using ManySpeech.AudioTagging;
using PreProcessUtils;

namespace ManySpeech.AudioTagging.Examples
{
    internal partial class OfflineAudioTagging : BaseTagging
    {
        public static OfflineTagging initOfflineOfflineTagging(string modelName)
        {
            string modelFilePath = applicationBase + "./" + modelName + "/model.onnx"; 
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            string configFilePath = "";// applicationBase + "./" + modelName + "/config.json";
            OfflineTagging offlineTagging = new OfflineTagging(modelFilePath: modelFilePath, tokensFilePath: tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
            return offlineTagging;
        }

        public static void OfflineOfflineTagging(List<float[]>? samples = null)
        {
            string modelName = "ced-mini-audio-tagging-onnx";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 15; i++)
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
            OfflineTagging offlineTagging = initOfflineOfflineTagging(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<OfflineStream> streams = new List<OfflineStream>();
            foreach (List<float[]> samplesListItem in samplesList)
            {
                OfflineStream stream = offlineTagging.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            List<OfflineTaggingResultEntity> results_batch = offlineTagging.GetResults(streams);
            foreach (OfflineTaggingResultEntity result in results_batch)
            {
                Console.WriteLine(result.Tagging);
                Console.WriteLine($"Top k tokens: [{string.Join(",",result.Tokens)}]");
                Console.WriteLine("=== Top-k Prediction Results ===");
                // Traverse the Top-k results and format the output
                for (int k = 0; k < result.Tokens.Count; k++)
                {
                    string tagging = result.Taggings[k];
                    double prob = result.Probs[k];
                    Console.WriteLine($"Top{k + 1}: {tagging,-30} Probability: {prob:F4}");
                }
                Console.WriteLine("");
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
            Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
            Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
        }
    }
}
