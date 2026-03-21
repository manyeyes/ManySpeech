using ManySpeech.DolphinAsr.Model;
using ManySpeech.DolphinAsr;
using PreProcessUtils;

namespace ManySpeech.SpeechLid.Examples
{
    internal partial class OfflineDolphinAsrLanguageID : BaseLid
    {
        public static LanguageID initOfflineLanguageID(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json"; // conf.yaml
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            LanguageID languageID = new LanguageID(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, tokensFilePath: tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
            return languageID;
        }

        public static void OfflineLanguageID(List<float[]>? samples = null)
        {
            string modelName = "DolphinAsr-base-int8-onnx-opt";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 3; i++)
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
            LanguageID languageID = initOfflineLanguageID(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<OfflineStream> streams = new List<OfflineStream>();
            foreach (List<float[]> samplesListItem in samplesList)
            {
                DolphinAsr.OfflineStream stream = languageID.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            List<OfflineRecognizerResultEntity> results_batch = languageID.GetResults(streams);
            foreach (OfflineRecognizerResultEntity result in results_batch)
            {
                Console.WriteLine($"language:{result.Language}, region:{result.Region}");
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
