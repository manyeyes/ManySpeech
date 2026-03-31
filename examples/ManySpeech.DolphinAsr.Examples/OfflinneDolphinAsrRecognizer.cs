using ManySpeech.DolphinAsr.Model;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.DolphinAsr.Examples
{
    internal partial class OfflinneDolphinAsrRecognizer : BaseAsr
    {
        public static OfflineRecognizer initOfflineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json"; // or conf.yaml
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, tokensFilePath: tokensFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void OfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "DolphinAsr-base-int8-onnx-opt";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 2; i++)
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
                // Specify language and region. If not specified, it will automatically detect.
                //stream.Language= "zh";
                //stream.Region = "CN";
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            // one
            foreach (var stream in streams)
            {
                OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                Console.WriteLine(AEDEmojiHelper.ReplaceTagsWithEmpty2(TextHelper.RemoveQuoteAroundPunctuation(result.Text)));
                StringBuilder r= new StringBuilder();
                r.AppendLine("{");
                r.AppendLine($"\"text\": \"{result.Text}\",");
                if (result.Tokens?.Count > 0)
                {
                    r.AppendLine($"\"tokens\":[{string.Join(",", result.Tokens.Select(x => $"\"{x}\"").ToArray())}]");
                }
                r.AppendLine("}");
                Console.WriteLine(r.ToString());
                Console.WriteLine("");
            }
            // batch
            //List<OfflineRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            //foreach (OfflineRecognizerResultEntity result in results_batch)
            //{
            //    Console.WriteLine(result.Text);
            //    Console.WriteLine(string.Join(",",result.Tokens));
            //    Console.WriteLine("");
            //}
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
            Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
            Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
        }
    }
}
