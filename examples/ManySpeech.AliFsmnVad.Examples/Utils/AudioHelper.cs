using NAudio.Wave;
using System.Diagnostics;

/// <summary>
/// audio processing
/// Copyright (c)  2023 by manyeyes
/// </summary>
namespace ManySpeech.AliFsmnVad.Examples.Utils
{
    public class AudioHelper
    {
        public static float[] GetFileSample(string wavFilePath, ref TimeSpan duration)
        {
            if (!File.Exists(wavFilePath))
            {
                return new float[1];
            }
            AudioFileReader audioFileReader = new AudioFileReader(wavFilePath);
            int sampleRate = audioFileReader.WaveFormat.SampleRate;
            int sourceChannels = audioFileReader.WaveFormat.Channels;
            byte[] datas = new byte[audioFileReader.Length];
            audioFileReader.ReadExactly(datas, 0, datas.Length);
            duration = audioFileReader.TotalTime;
            float[] wavsdata = new float[datas.Length / sizeof(float)];
            Buffer.BlockCopy(datas, 0, wavsdata, 0, datas.Length);
            if (sampleRate != 16000)
            {
                wavsdata = Resample(wavsdata, sampleRate, 16000, sourceChannels: sourceChannels);
            }
            return wavsdata;
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
                    int channels = 1;// OLDfmt.Channels;
                    int bitsPerSample = 32;
                    int newSampleRate = 16000;
                    var targetFormat = new WaveFormat(newSampleRate, bitsPerSample, channels);
                    using (var _mediaFileReaderResampler = new MediaFoundationResampler(_mediaFileReader, targetFormat))
                    {
                        _mediaFileReaderResampler.ResamplerQuality = 60; // 设置重采样质量 (0-100)
                        int bytesPerFrame = _mediaFileReaderResampler.WaveFormat.BitsPerSample / 8 * OLDfmt.Channels;
                        int bufferedFrames = (int)Math.Ceiling(_mediaFileReader.Length / bytesPerFrame
                            * ((float)targetFormat.SampleRate / (float)OLDfmt.SampleRate)
                            * ((float)targetFormat.Channels / OLDfmt.Channels)
                            * (targetFormat.BitsPerSample / OLDfmt.BitsPerSample));
                        ISampleProvider? samples = _mediaFileReaderResampler.ToSampleProvider();
                        wavdata = new float[bufferedFrames * bytesPerFrame / sizeof(float)];
                        samples.Read(wavdata, 0, wavdata.Length);
                        duration = _mediaFileReader.TotalTime;
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception("Get media sample failed", ex);
            }
            return wavdata;
        }

        public static List<float[]> GetFileChunkSamples(string wavFilePath, ref TimeSpan duration, int chunkSize = 160 * 6 * 10)
        {
            List<float[]> wavdatas = new List<float[]>();
            if (!File.Exists(wavFilePath))
            {
                Trace.Assert(File.Exists(wavFilePath), "file does not exist:" + wavFilePath);
                wavdatas.Add(new float[1]);
                return wavdatas;
            }
            AudioFileReader audioFileReader = new AudioFileReader(wavFilePath);
            int sampleRate = audioFileReader.WaveFormat.SampleRate;
            int sourceChannels = audioFileReader.WaveFormat.Channels;
            byte[] datas = new byte[audioFileReader.Length];
            audioFileReader.ReadExactly(datas);
            duration = audioFileReader.TotalTime;
            float[] wavsdata = new float[datas.Length / sizeof(float)];
            Buffer.BlockCopy(datas, 0, wavsdata, 0, datas.Length);
            if (sampleRate != 16000)
            {
                wavsdata = Resample(wavsdata, sampleRate, 16000, sourceChannels: sourceChannels);
            }
            int wavsLength = wavsdata.Length;
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
                float[] wavdata = new float[dataCount];
                Array.Copy(wavsdata, offset, wavdata, 0, dataCount);
                wavdatas.Add(wavdata);
            }
            return wavdatas;
        }

        public static List<float[]> GetMediaChunkSamples(string mediaFilePath, ref TimeSpan duration, int chunkSize = 160 * 6 * 10)
        {
            List<float[]> wavdatas = new List<float[]>();
            List<TimeSpan> durations = new List<TimeSpan>();
            if (!File.Exists(mediaFilePath))
            {
                Trace.Assert(File.Exists(mediaFilePath), "file does not exist:" + mediaFilePath);
                wavdatas.Add(new float[1]);
                return wavdatas;
            }
            using (MediaFoundationReader _mediaFileReader = new MediaFoundationReader(mediaFilePath))
            {
                var OLDfmt = _mediaFileReader.WaveFormat;
                var ieeeFloatWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(16000, 1); // mono
                using (var _mediaFileReaderResampler = new MediaFoundationResampler(_mediaFileReader, ieeeFloatWaveFormat))
                {
                    int bytesPerFrame = _mediaFileReader.WaveFormat.BitsPerSample / 8 * _mediaFileReader.WaveFormat.Channels; // 100 millsec 1 frames
                    int bufferedFrames = (int)Math.Ceiling(_mediaFileReader.Length / bytesPerFrame * ((float)ieeeFloatWaveFormat.SampleRate / (float)OLDfmt.SampleRate / OLDfmt.Channels));
                    ISampleProvider? _samples = _mediaFileReaderResampler.ToSampleProvider();
                    int chunkNum = (int)Math.Floor((double)bufferedFrames / chunkSize);
                    for (int i = 0; i < chunkNum; i++)
                    {
                        int offset = 0;
                        int dataCount = 0;
                        if (Math.Abs(bufferedFrames - i * chunkSize) > chunkSize)
                        {
                            offset = i * chunkSize;
                            dataCount = chunkSize;
                        }
                        else
                        {
                            offset = i * chunkSize;
                            dataCount = bufferedFrames - i * chunkSize;
                        }
                        float[] _frames = new float[dataCount];
                        if (i >= chunkNum - 1)
                        {
                            _frames = new float[chunkSize];
                        }
                        _samples.Read(_frames, 0, dataCount);

                        //_mediaFileReader.Read(datas, offset, dataCount);
                        TimeSpan curDuration = TimeSpan.FromMilliseconds(dataCount / 2 * 1000 / 16000);
                        durations.Add(curDuration);
                        wavdatas.Add(_frames);
                    }
                }
            }
            duration = durations.Aggregate(TimeSpan.Zero, (currentTotal, nextDuration) => currentTotal + nextDuration);
            return wavdatas;
        }
        /// <summary>
        /// Resamples audio sample rate
        /// </summary>
        /// <param name="sourceData">Source audio data (PCM format, range [-1, 1])</param>
        /// <param name="sourceSampleRate">Source sample rate (e.g., 44100)</param>
        /// <param name="targetSampleRate">Target sample rate (e.g., 16000)</param>
        /// <returns>Resampled audio data</returns>
        public static float[] Resample(float[] sourceData, int sourceSampleRate, int targetSampleRate)
        {
            if (sourceSampleRate <= 0 || targetSampleRate <= 0)
            {
                throw new ArgumentException("Sample rates must be positive numbers");
            }
            if (sourceData == null || sourceData.Length == 0)
            {
                return Array.Empty<float>();
            }
            // Calculate sample rate ratio
            double ratio = (double)sourceSampleRate / targetSampleRate;
            // Calculate target data length
            int targetLength = (int)Math.Round(sourceData.Length / ratio);
            float[] targetData = new float[targetLength];
            // Calculate new sample points using linear interpolation
            for (int i = 0; i < targetLength; i++)
            {
                // Calculate corresponding position in source data (may be a decimal)
                double sourcePosition = i * ratio;
                // Integer part (source data index)
                int index = (int)sourcePosition;
                // Decimal part (interpolation weight)
                double fraction = sourcePosition - index;
                // Handle boundary cases (avoid out-of-bounds)
                if (index >= sourceData.Length - 1)
                {
                    targetData[i] = sourceData[sourceData.Length - 1];
                    continue;
                }
                // Linear interpolation: f(x) = (1 - fraction) * x0 + fraction * x1
                targetData[i] = (float)((1 - fraction) * sourceData[index] + fraction * sourceData[index + 1]);
            }
            return targetData;
        }

        public static float[] Resample(float[] sourceData, int sourceSampleRate, int targetSampleRate, int sourceChannels = 1)
        {
            // 参数验证
            if (sourceSampleRate <= 0 || targetSampleRate <= 0)
            {
                throw new ArgumentException("采样率必须为正数", nameof(sourceSampleRate));
            }
            if (sourceChannels != 1 && sourceChannels != 2)
            {
                throw new ArgumentException("仅支持单通道（1）或双通道（2）输入", nameof(sourceChannels));
            }
            if (sourceData == null || sourceData.Length == 0)
            {
                return Array.Empty<float>();
            }

            // 步骤1：将双通道转为单通道（如果需要）
            float[] monoData = sourceData;
            if (sourceChannels == 2)
            {
                // 计算单通道数据长度（双通道数据长度应为偶数，若为奇数则忽略最后一个字节）
                int monoLength = sourceData.Length / 2;
                monoData = new float[monoLength];

                // 合并左右声道（取平均值）
                for (int i = 0; i < monoLength; i++)
                {
                    float leftChannel = sourceData[i * 2];       // 左声道数据
                    float rightChannel = sourceData[i * 2 + 1];  // 右声道数据
                    monoData[i] = (leftChannel + rightChannel) * 0.5f;  // 平均合并
                }
            }

            // 步骤2：执行重采样（复用原有逻辑，基于单通道数据处理）
            double ratio = (double)sourceSampleRate / targetSampleRate;
            int targetLength = (int)Math.Round(monoData.Length / ratio);
            float[] targetData = new float[targetLength];

            for (int i = 0; i < targetLength; i++)
            {
                double sourcePosition = i * ratio;
                int index = (int)sourcePosition;
                double fraction = sourcePosition - index;

                // 处理边界情况（避免索引越界）
                if (index >= monoData.Length - 1)
                {
                    targetData[i] = monoData[monoData.Length - 1];
                    continue;
                }

                // 线性插值计算重采样后的值
                targetData[i] = (float)((1 - fraction) * monoData[index] + fraction * monoData[index + 1]);
            }

            return targetData;
        }

        public static string SaveSampleToWavFile(float[]? sample, ref TimeSpan duration, string wavFileDir = "temp", string? stemName = null, int sampleRate = 16000, int channels = 1, bool isResample = false)
        {
            if (string.IsNullOrEmpty(wavFileDir))
            {
                return "未指定目录";
            }
            if (!Directory.Exists(wavFileDir))
            {
                Directory.CreateDirectory(wavFileDir);
            }
            Guid guid = Guid.NewGuid();
            string wavFileName = string.IsNullOrEmpty(stemName) ? guid.ToString() : stemName + ".wav";
            string wavFileFullName = wavFileDir + wavFileName;
            try
            {
                if (sample != null && sample.Length > 0)
                {
                    byte[] datas = new byte[sizeof(float) * sample.Length];
                    System.Buffer.BlockCopy(sample, 0, datas, 0, datas.Length);
                    using (var memoryStream = new MemoryStream(datas))
                    {
                        var OLDfmt = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
                        var ieeeFloatWaveFormat = OLDfmt;
                        if ((sampleRate != 16000 || channels != 1) && isResample)
                        {
                            sample = Resample(sample, sampleRate, 16000, sourceChannels: channels);
                            ieeeFloatWaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(16000, 1);
                        }
                        using (WaveFileWriter writer = new WaveFileWriter(wavFileFullName, ieeeFloatWaveFormat))
                        {
                            writer.WriteSamples(sample, 0, sample.Length);
                            writer.Flush();
                        }
                    }
                }
            }
            catch (Exception ex)
            {

            }
            return wavFileFullName;
        }

        /// <summary>
        /// 通过文件头特征判断是否为音频文件
        /// </summary>
        public static bool IsAudioByHeader(string filePath)
        {
            if (!File.Exists(filePath))
                return false;

            // 读取文件头前16字节（足够判断常见音频类型）
            byte[] header = new byte[16];
            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                int bytesRead = stream.Read(header, 0, header.Length);
                if (bytesRead < header.Length)
                    return false; // 文件过小，无法判断
            }

            // 检查是否为常见音频文件的文件头
            return IsMp3Header(header) ||
                   IsWavHeader(header) ||
                   IsOggHeader(header) ||
                   IsAviHeader(header) ||
                   IsMp4Header(header) ||
                   IsFlacHeader(header);
        }

        // 判断是否为MP3文件头（ID3v2或帧头）
        private static bool IsMp3Header(byte[] header)
        {
            // ID3v2标记：前3字节为 'I','D','3'
            if (header[0] == 0x49 && header[1] == 0x44 && header[2] == 0x33)
                return true;

            // MP3帧头特征（简化判断）
            // 帧头前11位为11111111111，第12位为0或1
            if (header.Length >= 4)
            {
                byte b1 = header[0];
                byte b2 = header[1];
                bool isFrameHeader = (b1 == 0xFF) && ((b2 & 0xE0) == 0xE0); // 0xE0 = 11100000
                return isFrameHeader;
            }
            return false;
        }

        // 判断是否为WAV文件头（RIFF + WAVE）
        private static bool IsWavHeader(byte[] header)
        {
            // WAV文件头前8字节：'R','I','F','F', 长度, 'W','A','V','E'
            if (header.Length >= 12)
            {
                bool isRiff = header[0] == 0x52 && header[1] == 0x49 &&
                              header[2] == 0x46 && header[3] == 0x46; // "RIFF"
                bool isWave = header[8] == 0x57 && header[9] == 0x41 &&
                              header[10] == 0x56 && header[11] == 0x45; // "WAVE"
                return isRiff && isWave;
            }
            return false;
        }

        // 判断是否为OGG文件头（OggS）
        private static bool IsOggHeader(byte[] header)
        {
            // OGG前4字节：'O','g','g','S'
            return header[0] == 0x4F && header[1] == 0x67 &&
                   header[2] == 0x67 && header[3] == 0x53;
        }

        // 判断是否为FLAC文件头（fLaC）
        private static bool IsFlacHeader(byte[] header)
        {
            // FLAC前4字节：'f','L','a','C'
            return header[0] == 0x66 && header[1] == 0x4C &&
                   header[2] == 0x61 && header[3] == 0x43;
        }

        /// <summary>
        /// 判断是否为MP4文件头
        /// MP4文件通常以"ftyp"作为文件标识
        /// </summary>
        public static bool IsMp4Header(byte[] header)
        {
            // 1. 处理null和长度不足的情况
            if (header == null || header.Length < 8) // 至少需要8字节（4字节大小 + 4字节"ftyp"）
                return false;

            // 2. 检查"ftyp"原子标识（第4-7字节，因为前4字节是原子大小）
            // 注意：标准"ftyp"原子通常位于文件开头，即前8字节为 [大小][ftyp]
            bool isFtypAtom = (header[4] == 0x66 && header[5] == 0x74 &&
                               header[6] == 0x79 && header[7] == 0x70);

            // 3. 兼容部分以"ftyp"直接开头的非标准文件（前4字节即为"ftyp"）
            bool isFtypAtStart = (header[0] == 0x66 && header[1] == 0x74 &&
                                  header[2] == 0x79 && header[3] == 0x70);

            return isFtypAtom || isFtypAtStart;
        }

        /// <summary>
        /// 判断是否为AVI文件头
        /// AVI文件以"RIFF"开头，且后续包含"AVI "标识
        /// </summary>
        public static bool IsAviHeader(byte[] header)
        {
            // AVI文件头特征：
            // 1. 前4字节为 "RIFF"（0x52494646）
            // 2. 第8-11字节为 "AVI "（0x41564920，注意末尾有空格）
            if (header.Length >= 12)
            {
                bool isRiff = header[0] == 0x52 &&  // 'R'
                              header[1] == 0x49 &&  // 'I'
                              header[2] == 0x46 &&  // 'F'
                              header[3] == 0x46;    // 'F'

                bool isAvi = header[8] == 0x41 &&   // 'A'
                             header[9] == 0x56 &&   // 'V'
                             header[10] == 0x49 &&  // 'I'
                             header[11] == 0x20;    // ' '（空格）

                return isRiff && isAvi;
            }
            return false;
        }
    }
}
