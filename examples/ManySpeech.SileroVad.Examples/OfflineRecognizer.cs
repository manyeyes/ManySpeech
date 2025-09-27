using PreProcessUtils;
using ManySpeech.AliParaformerAsr;

namespace ManySpeech.SileroVad.Examples
{
    internal static partial class Program
    {
        private static OfflineRecognizer _offlineRecognizer;
        public static OfflineRecognizer initOfflineRecognizer(string modelName)
        {
            if (_offlineRecognizer == null)
            {
                TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
                string configFilePath = applicationBase + "./" + modelName + "/asr.josn";
                string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
                string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
                _offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath, tokensFilePath: tokensFilePath);
                TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                double elapsed_milliseconds_init = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                Console.WriteLine("loading asr model elapsed_milliseconds:{0}", elapsed_milliseconds_init.ToString());
            }
            return _offlineRecognizer;
        }
        public static void OfflineRecognizer(List<float[]>? samples = null)
        {
            string modelName = "speech_paraformer-large-zh-cn-16k-timestamp-vocab8404-onnx";
            OfflineRecognizer offlineRecognizer = initOfflineRecognizer(modelName);
            TimeSpan total_duration = new TimeSpan(0L);
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int i = 0; i < 2; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        break;
                    }
                    TimeSpan duration = TimeSpan.Zero;
                    //supports Windows, Mac, and Linux
                    //float[] sample = AudioHelper.GetFileSamples(wavFilePath: wavFilePath,ref duration);
                    //supports Windows only
                    float[]? sample = AudioHelper.GetMediaSample(mediaFilePath: wavFilePath, ref duration);
                    if(sample != null)
                    {
                        samples.Add(sample);
                        total_duration += duration;
                    }
                }
            }
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
            foreach (var sample in samples)
            {
                AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                stream.AddSamples(sample);
                streams.Add(stream);
            }
            List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
            foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
            {
                if (!string.IsNullOrEmpty(result.Text))
                {
                    Console.WriteLine(result.Text);
                }
                //for (int i = 0; i < result.Tokens.Count; i++)
                //{
                //    Console.WriteLine(string.Format("{0}:[{1},{2}]", result.Tokens[i], result.Timestamps[i].First(), result.Timestamps[i].Last()));
                //}
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            //double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            //double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            //Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            //Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            //Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            //Console.WriteLine("Hello, World!");
        }
    }
}
