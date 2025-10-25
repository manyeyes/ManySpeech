using ManySpeech.WhisperAsr;
using ManySpeech.WhisperAsr.Model;
using PreProcessUtils;

namespace ManySpeech.WhisperAsr.Examples
{
    internal static partial class Program
    {
        public static TranscribeRecognizer initWhisperAsrTranscribeRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
            string configFilePath = applicationBase + "./" + modelName + "/conf.json";
            WhisperAsr.TranscribeRecognizer transcribeRecognizer = new WhisperAsr.TranscribeRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, threadsNum: 1);
            return transcribeRecognizer;
        }

        public static void test_WhisperAsrTranscribeRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "whisper-tiny-onnx";
            //string modelName = "whisper-tiny-en-onnx";
            //string modelName = "whisper-base-onnx";
            //string modelName = "whisper-base-en-onnx";
            string modelName = "whisper-small-onnx";
            //string modelName = "whisper-small-en-onnx";
            //string modelName = "whisper-medium-onnx";
            //string modelName = "whisper-medium-en-onnx";
            //string modelName = "whisper-large-v1-onnx";
            //string modelName = "whisper-large-v2-onnx";
            //string modelName = "whisper-large-v3-finetune-onnx";
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
                    ////float[] sample = AudioHelper.GetMediaSample(wavFilePath, ref duration);
                    //float[] sample = AudioHelper.GetFileSample(wavFilePath, ref duration);
                    //samples.Add(sample);
                    //totalDuration += duration;
                    //method 2
                    TimeSpan duration = TimeSpan.Zero;
                    //float[] sample = AudioHelper.GetMediaSample(wavFilePath, ref duration);
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
            TranscribeRecognizer offlineRecognizer = initWhisperAsrTranscribeRecognizer(modelName);
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<TranscribeStream> streams = new List<TranscribeStream>();
            // method 1
            //foreach (var sample in samples)
            //{
            //    TranscribeStream stream = offlineRecognizer.CreateTranscribeStream();
            //    stream.AddSamples(sample);
            //    streams.Add(stream);
            //}
            // method 2
            foreach (List<float[]> samplesListItem in samplesList)
            {
                TranscribeStream stream = offlineRecognizer.CreateTranscribeStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            // decode,fit batch=1
            //foreach (WhisperAsr.TranscribeStream stream in streams)
            //{
            //    WhisperAsr.Model.TranscribeRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
            //    Console.WriteLine(result.Text);
            //    Console.WriteLine("");
            //}
            //fit batch>1,but all in one
            List<TranscribeRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            foreach (TranscribeRecognizerResultEntity result in results_batch)
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
