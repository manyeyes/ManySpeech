using NAudio.Wave;
using System.Diagnostics;

namespace ManySpeech.MoonshineAsr.Examples.Utils
{
    public class AudioHelper
    {
        public static float[] GetFileSample(string wavFilePath, ref TimeSpan duration, bool normalize = true)
        {
            if (!File.Exists(wavFilePath))
            {
                return new float[1];
            }
            AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
            byte[] datas = new byte[_audioFileReader.Length];
            _audioFileReader.Read(datas, 0, datas.Length);
            duration = _audioFileReader.TotalTime;
            float[] sample = new float[datas.Length / sizeof(float)];
            Buffer.BlockCopy(datas, 0, sample, 0, datas.Length);
            if (!normalize)
            {
                sample = sample.Select(x => (float)Float32ToInt16(x)).ToArray();
            }
            return sample;
        }        

        public static List<float[]> GetFileChunkSamples(string wavFilePath, ref TimeSpan duration, int chunkSize = 160 * 6 * 10, bool normalize = true)
        {
            List<float[]> wavdatas = new List<float[]>();
            if (!File.Exists(wavFilePath))
            {
                Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
                wavdatas.Add(new float[1]);
                return wavdatas;
            }
            AudioFileReader _audioFileReader = new AudioFileReader(wavFilePath);
            byte[] datas = new byte[_audioFileReader.Length];
            _audioFileReader.Read(datas);
            duration = _audioFileReader.TotalTime;
            float[] wavsdata = new float[datas.Length / sizeof(float)];
            int wavsLength = wavsdata.Length;
            Buffer.BlockCopy(datas, 0, wavsdata, 0, datas.Length);
            int chunkNum = (int)Math.Ceiling((double)wavsLength / chunkSize);
            for (int i = 0; i < chunkNum; i++)
            {
                int offset;
                int dataCount;
                if (Math.Abs(wavsLength - i * chunkSize) > chunkSize)
                {
                    offset = i * chunkSize;
                    dataCount = chunkSize;
                }
                else
                {
                    offset = i * chunkSize;
                    dataCount = wavsLength - i * chunkSize;
                }
                float[] wavdata = new float[dataCount];//dataCount
                Array.Copy(wavsdata, offset, wavdata, 0, dataCount);
                if (!normalize)
                {
                    wavdata = wavdata.Select(x => (float)Float32ToInt16(x)).ToArray();
                }
                wavdatas.Add(wavdata);

            }
            return wavdatas;
        }

        private static Int16 Float32ToInt16(float sample)
        {
            if (sample < -0.999999f)
            {
                return Int16.MinValue;
            }
            else if (sample > 0.999999f)
            {
                return Int16.MaxValue;
            }
            else
            {
                if (sample < 0)
                {
                    return (Int16)(Math.Floor(sample * 32767.0f));
                }
                else
                {
                    return (Int16)(Math.Ceiling(sample * 32767.0f));
                }
            }
        }

        /// <summary>
        /// get media sample
        /// supports Windows only
        /// </summary>
        /// <param name="mediaFilePath"></param>
        /// <param name="duration"></param>
        /// <returns></returns>
        public static float[]? GetMediaSample(string mediaFilePath, ref TimeSpan duration)
        {
            float[]? wavdata = null;
            try
            {
                if (!File.Exists(mediaFilePath))
                {
                    Trace.Assert(File.Exists(mediaFilePath), "file does not exist:" + mediaFilePath);
                    return wavdata;
                }
                using (MediaFoundationReader _mediaFileReader = new MediaFoundationReader(mediaFilePath))
                {
                    WaveFormat OLDfmt = _mediaFileReader.WaveFormat;
                    int newSampleRate = 16000;
                    var ieeeFloatWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(16000, 1); // mono
                    using (var _mediaFileReaderResampler = new MediaFoundationResampler(_mediaFileReader, ieeeFloatWaveFormat))
                    {
                        int bytesPerFrame = _mediaFileReader.WaveFormat.BitsPerSample / 8 * _mediaFileReader.WaveFormat.Channels;
                        int bufferedFrames = (int)Math.Ceiling(_mediaFileReader.Length / bytesPerFrame * ((float)ieeeFloatWaveFormat.SampleRate / (float)OLDfmt.SampleRate));
                        ISampleProvider? samples = _mediaFileReaderResampler.ToSampleProvider();
                        float[] _frames = new float[bufferedFrames];
                        samples.Read(_frames, 0, _frames.Length);
                        wavdata = _frames;
                        duration = _mediaFileReader.TotalTime;
                    }
                }
            }
            catch (Exception ex)
            {
                //
            }
            return wavdata;
        }
    }
}
