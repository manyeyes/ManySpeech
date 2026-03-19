using ManySpeech.DolphinAsr.Model;
using PreProcessUtils;
using System.Text;
using System.Text.RegularExpressions;

namespace ManySpeech.DolphinAsr.Examples
{
    internal partial class OfflinneDolphinAsrRecognizer : BaseAsr
    {
        public static OfflineRecognizer initOfflineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json"; // or conf.yaml
            string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, tokensFilePath: tokensFilePath, threadsNum: 16);
            return offlineRecognizer;
        }

        public static void OfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "DolphinAsr-base-onnx";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 1; i++)
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
                Console.WriteLine(RemoveAngleBracketContent(RemoveQuoteAroundPunctuation(result.Text)));
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
        public static string RemoveQuoteAroundPunctuation(string input)
        {
            // 正则表达式：匹配单引号包裹的中文标点
            string pattern = @"'([,，.。!！?？;；:：""''（）[\]【】<>《》/\、·])'";
            // 替换为捕获到的标点本身
            return Regex.Replace(input, pattern, "$1");
        }
        public static string RemoveAngleBracketContent(string input)
        {
            if (string.IsNullOrEmpty(input))
            {
                return input;
            }

            // 核心正则表达式：匹配 < 开头、> 结尾的任意字符（非贪婪匹配）
            // \< 转义匹配 <，\> 转义匹配 >，.*? 非贪婪匹配中间任意字符
            string pattern = @"\<.*?\>";
            return Regex.Replace(input, pattern, string.Empty, RegexOptions.Compiled);
        }

    }
}
