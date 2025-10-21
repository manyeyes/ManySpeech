using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AudioInOut.Base;

namespace AudioInOut.Recorder
{
    public class LinuxAlsaRecorder : BaseRecorder, IDisposable
    {
        private nint _pcmHandle = default(nint);//nint.Zero;
        private bool _isCapturing = false;
        private Thread? _captureThread;
        private readonly ConcurrentQueue<float[]> _audioChunkQueue;
        private readonly int _bufferMilliseconds;

        public bool IsCapturing => _isCapturing;
        public const int SampleRate = 16000;
        public const int BitsPerSample = 16;
        public const int Channels = 1;

        public LinuxAlsaRecorder(int bufferMilliseconds = 100)
        {
            _bufferMilliseconds = bufferMilliseconds;
            _audioChunkQueue = new ConcurrentQueue<float[]>();
        }

        public override async Task StartCapture()
        {
            if (_isCapturing) return;

            try
            {
                // 打开默认PCM设备
                int result = AlsaNative.snd_pcm_open(out _pcmHandle, "default", AlsaNative.SND_PCM_STREAM_CAPTURE, 0);
                if (result < 0)
                {
                    throw new InvalidOperationException($"ALSA open failed: {AlsaNative.GetErrorString(result)}");
                }

                // 设置硬件参数
                SetHardwareParameters();

                _isCapturing = true;
                _captureThread = new Thread(CaptureThread)
                {
                    IsBackground = true,
                    Priority = ThreadPriority.AboveNormal
                };
                _captureThread.Start();

                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Linux ALSA 麦克风采集已启动");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ALSA start failed: {ex.Message}");
                throw;
            }
        }

        private void SetHardwareParameters()
        {
            nint hwParams = default(nint);//nint.Zero;

            try
            {
                // 分配硬件参数结构体
                AlsaNative.snd_pcm_hw_params_malloc(ref hwParams);
                AlsaNative.snd_pcm_hw_params_any(_pcmHandle, hwParams);

                // 设置访问类型
                AlsaNative.snd_pcm_hw_params_set_access(_pcmHandle, hwParams, AlsaNative.SND_PCM_ACCESS_RW_INTERLEAVED);

                // 设置格式
                AlsaNative.snd_pcm_hw_params_set_format(_pcmHandle, hwParams, AlsaNative.SND_PCM_FORMAT_S16_LE);

                // 设置采样率
                uint sampleRate = SampleRate;
                AlsaNative.snd_pcm_hw_params_set_rate_near(_pcmHandle, hwParams, ref sampleRate, default(nint));//nint.Zero;

                // 设置通道数
                AlsaNative.snd_pcm_hw_params_set_channels(_pcmHandle, hwParams, Channels);

                // 设置缓冲区大小
                uint bufferTime = (uint)(_bufferMilliseconds * 1000); // 微秒
                AlsaNative.snd_pcm_hw_params_set_buffer_time_near(_pcmHandle, hwParams, ref bufferTime, default(nint));//nint.Zero;

                // 应用参数
                int result = AlsaNative.snd_pcm_hw_params(_pcmHandle, hwParams);
                if (result < 0)
                {
                    throw new InvalidOperationException($"ALSA set params failed: {AlsaNative.GetErrorString(result)}");
                }
            }
            finally
            {
                if (hwParams != default(nint))//nint.Zero;
                {
                    AlsaNative.snd_pcm_hw_params_free(hwParams);
                }
            }
        }

        private void CaptureThread()
        {
            int bufferSize = SampleRate * _bufferMilliseconds * (BitsPerSample / 8) * Channels / 1000;
            byte[] buffer = new byte[bufferSize];

            while (_isCapturing && _pcmHandle != default(nint))//nint.Zero;
            {
                try
                {
                    // 读取音频数据
                    int framesRead = AlsaNative.snd_pcm_readi(_pcmHandle, buffer, bufferSize / (BitsPerSample / 8));

                    if (framesRead < 0)
                    {
                        // 处理错误或中断
                        framesRead = AlsaNative.snd_pcm_recover(_pcmHandle, framesRead, 0);
                        if (framesRead < 0)
                        {
                            Console.WriteLine($"ALSA read error: {AlsaNative.GetErrorString(framesRead)}");
                            Thread.Sleep(10);
                            continue;
                        }
                    }

                    if (framesRead > 0)
                    {
                        int bytesRead = framesRead * (BitsPerSample / 8) * Channels;
                        float[] normalizedSamples = ConvertToFloatSamples(buffer, bytesRead);
                        _audioChunkQueue.Enqueue(normalizedSamples);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Capture thread error: {ex.Message}");
                    Thread.Sleep(10);
                }
            }
        }

        private float[] ConvertToFloatSamples(byte[] buffer, int bytesRecorded)
        {
            int sampleCount = bytesRecorded / (BitsPerSample / 8);
            float[] normalizedSamples = new float[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                int byteIndex = i * 2;
                if (byteIndex + 1 < bytesRecorded)
                {
                    short pcmValue = BitConverter.ToInt16(buffer, byteIndex);
                    normalizedSamples[i] = pcmValue / 32768.0f;
                }
            }

            return normalizedSamples;
        }

        public override void StopCapture()
        {
            _isCapturing = false;
            _captureThread?.Join(1000);
            _captureThread = null;

            if (_pcmHandle != default(nint))//nint.Zero;
            {
                AlsaNative.snd_pcm_close(_pcmHandle);
                _pcmHandle = default(nint);//nint.Zero;
            }

            _audioChunkQueue.Enqueue(null);
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Linux ALSA 麦克风采集已停止");
        }

        public override async Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken)
        {
            while (_isCapturing && _audioChunkQueue.IsEmpty)
            {
                if (cancellationToken.IsCancellationRequested) return null;
                await Task.Delay(10, cancellationToken);
            }

            if (!_audioChunkQueue.TryDequeue(out float[]? chunk) || cancellationToken.IsCancellationRequested)
            {
                return null;
            }

            return chunk == null ? null : new List<List<float[]>> { new List<float[]> { chunk } };
        }

        //public override void Dispose()
        //{
        //    StopCapture();
        //}
    }

    // ALSA Native 互操作
    internal static class AlsaNative
    {
        public const int SND_PCM_STREAM_CAPTURE = 1;
        public const int SND_PCM_ACCESS_RW_INTERLEAVED = 3;
        public const int SND_PCM_FORMAT_S16_LE = 2;

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_open(out nint pcm, string name, int stream, int mode);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_close(nint pcm);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_readi(nint pcm, byte[] buffer, nint size);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_recover(nint pcm, int err, int silent);

        [DllImport("libasound.so.2")]
        public static extern void snd_pcm_hw_params_malloc(ref nint ptr);

        [DllImport("libasound.so.2")]
        public static extern void snd_pcm_hw_params_free(nint ptr);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_any(nint pcm, nint params_ptr);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_set_access(nint pcm, nint params_ptr, int access);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_set_format(nint pcm, nint params_ptr, int format);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_set_rate_near(nint pcm, nint params_ptr, ref uint val, nint dir);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_set_channels(nint pcm, nint params_ptr, uint val);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params_set_buffer_time_near(nint pcm, nint params_ptr, ref uint val, nint dir);

        [DllImport("libasound.so.2")]
        public static extern int snd_pcm_hw_params(nint pcm, nint params_ptr);

        [DllImport("libasound.so.2")]
        public static extern nint snd_strerror(int errnum);

        public static string GetErrorString(int errorCode)
        {
            nint ptr = snd_strerror(errorCode);
            return Marshal.PtrToStringAnsi(ptr) ?? $"ALSA Error {errorCode}";
        }
    }
}

