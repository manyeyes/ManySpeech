using ManySpeech.WhisperAsr.Model;
using ManySpeech.WhisperAsr;
using PreProcessUtils;

namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        public static OfflineRecognizer initWhisperAsrOfflineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void test_WhisperAsrOfflineRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "whisper-tiny-onnx";
            //string modelName = "whisper-tiny-en-onnx";
            //string modelName = "whisper-base-onnx";
            //string modelName = "whisper-base-en-onnx";
            string modelName = "whisper-small-onnx";
            //string modelName = "whisper-small-en-onnx";
            //string modelName = "whisper-small-cantonese-onnx";
            //string modelName = "whisper-medium-onnx";
            //string modelName = "whisper-medium-en-onnx";
            //string modelName = "whisper-large-v1-onnx";
            //string modelName = "whisper-large-v2-onnx";
            //string modelName = "whisper-large-v3-onnx";
            //string modelName = "whisper-large-v3-turbo-onnx";
            //string modelName = "whisper-large-v3-turbo-zh-onnx";
            //string modelName = "distil-whisper-small-en-onnx";
            //string modelName = "distil-whisper-medium-en-onnx";
            //string modelName = "distil-whisper-large-v2-en-onnx";
            //string modelName = "distil-whisper-large-v3-en-onnx";
            //string modelName = "distil-whipser-large-v3.5-en-onnx";
            //string modelName = "distil-whisper-large-v2-multi-hans-onnx";
            //string modelName = "distil-whisper-small-cantonese-onnx-alvanlii-20240404";
            TimeSpan totalDuration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                for (int i = 0; i < 1; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./test_wavs/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    //TimeSpan duration = TimeSpan.Zero;
                    //float[] sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
                    //samples.Add(sample);
                    //totalDuration += duration;
                    //method 2
                    TimeSpan duration = TimeSpan.Zero;
                    //float[] sample=AudioHelper.GetMediaSample(wavFilePath, ref duration);
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
            OfflineRecognizer offlineRecognizer = initWhisperAsrOfflineRecognizer(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);            
            List<OfflineStream> streams = new List<OfflineStream>();
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
                OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            // decode,fit batch=1
            //foreach (WhisperAsr.OfflineStream stream in streams)
            //{
            //    WhisperAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
            //    Console.WriteLine(result.Text);
            //    Console.WriteLine("");
            //}
            //fit batch>1,but all in one
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
