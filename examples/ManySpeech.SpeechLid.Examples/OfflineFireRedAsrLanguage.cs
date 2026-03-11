using ManySpeech.FireRedAsr.Model;
using ManySpeech.FireRedAsr;
using PreProcessUtils;

namespace ManySpeech.SpeechLid.Examples
{
    internal partial class OfflineFireRedAsrLanguage : BaseLid
    {
        public static LanguageDetection initFireRedAsrLanguageDetection(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            //string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            string configFilePath = "";
            string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            LanguageDetection languageDetection = new LanguageDetection(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
            return languageDetection;
        }

        public static void FireRedAsrLanguageDetection(List<float[]>? samples = null)
        {
            string modelName = "FireRedLID-int8-onnx";
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
            LanguageDetection languageDetection = initFireRedAsrLanguageDetection(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<OfflineStream> streams = new List<OfflineStream>();
            foreach (List<float[]> samplesListItem in samplesList)
            {
                FireRedAsr.OfflineStream stream = languageDetection.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            List<OfflineRecognizerResultEntity> results_batch = languageDetection.GetResults(streams);
            foreach (OfflineRecognizerResultEntity result in results_batch)
            {
                Console.WriteLine(result.Language);
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
