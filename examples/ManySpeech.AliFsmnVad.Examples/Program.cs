// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using ManySpeech.AliFsmnVad;
using ManySpeech.AliFsmnVad.Model;

internal static class Program
{
	[STAThread]
	private static void Main()
	{
		string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
		string modelName = "speech_fsmn_vad_zh-cn-16k-common-onnx";
		string modelFilePath = applicationBase + "./"+ modelName + "/model.onnx";
		string configFilePath = applicationBase + "./"+ modelName + "/vad.json";
		string mvnFilePath = applicationBase + "./"+ modelName + "/vad.mvn";
		int batchSize = 2;
		TimeSpan start_time0 = new TimeSpan(DateTime.Now.Ticks);
		FsmnVad aliFsmnVad = new FsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
		TimeSpan end_time0 = new TimeSpan(DateTime.Now.Ticks);
		double elapsed_milliseconds0 = end_time0.TotalMilliseconds - start_time0.TotalMilliseconds;
		Console.WriteLine("load model and init config elapsed_milliseconds:{0}", elapsed_milliseconds0.ToString());
		List<float[]> samples = new List<float[]>();
		TimeSpan total_duration = new TimeSpan(0L);
		for (int i = 0; i < 1; i++)
		{
			string wavFilePath = string.Format(applicationBase + "./"+ modelName + "/example/{0}.wav", i.ToString());
			if (!File.Exists(wavFilePath))
			{
				continue;
			}
            TimeSpan duration = TimeSpan.Zero;
            //supports Windows, Mac, and Linux
            //float[] sample = AudioHelper.GetFileSamples(wavFilePath: wavFilePath,ref duration);
            //supports Windows only
            float[]? sample = PreProcessUtils.AudioHelper.GetMediaSample(mediaFilePath: wavFilePath, ref duration);
            if (sample != null)
            {
                samples.Add(sample);
                total_duration += duration;
            }	
        }
		TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
		//SegmentEntity[] segments_duration = aliFsmnVad.GetSegments(samples);
		SegmentEntity[] segments_duration = aliFsmnVad.GetSegmentsByStep(samples);
		TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
		Console.WriteLine("vad infer result:");
		foreach (SegmentEntity segment in segments_duration)
		{
			Console.Write("[");
			foreach (var x in segment.Segment) 
			{
				Console.Write("[" + string.Join(",", x.ToArray()) + "]");
			}
			Console.Write("]\r\n");
		}

		double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
		double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
		Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
		Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
		Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
		Console.WriteLine("------------------------");
	}
}