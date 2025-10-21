using Android.Media;
using ManySpeech.Maui.Sample.Platforms.Android;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoding = Android.Media.Encoding;

[assembly: Dependency(typeof(AndroidAudioCaptureService))]
namespace ManySpeech.Maui.Sample.Platforms.Android
{
    internal class AndroidAudioCaptureService : AudioInOut.Base.BaseRecorder, IDisposable
    {
        private bool _disposed = false;
        private readonly object _disposeLock = new object();

        private AudioRecord _audioRecord;
        private readonly int _bufferSize;
        private readonly int _bufferMilliseconds;
        private CancellationTokenSource _cancellationTokenSource;
        private readonly ConcurrentQueue<float[]> _audioChunkQueue;
        private Task _readingTask;

        public bool IsCapturing { get; protected set; }

        // Android 音频参数
        public const int SampleRate = 16000;
        public const ChannelIn ChannelConfig = ChannelIn.Mono;
        public const Encoding AudioFormat = Encoding.Pcm16bit;
        public const int BitsPerSample = 16;
        public const int Channels = 1;

        public AndroidAudioCaptureService(int bufferMilliseconds = 100)
        {
            _bufferMilliseconds = bufferMilliseconds;
            _audioChunkQueue = new ConcurrentQueue<float[]>();

            // 计算缓冲区大小
            _bufferSize = AudioRecord.GetMinBufferSize(SampleRate, ChannelConfig, AudioFormat);

            // 检查缓冲区大小是否有效（GetMinBufferSize 返回负数表示错误）
            if (_bufferSize <= 0)
            {
                throw new InvalidOperationException($"不支持的音频参数，获取缓冲区大小失败: {_bufferSize}");
            }

            // 确保缓冲区大小合理，基于指定的毫秒数
            int desiredBufferSize = SampleRate * _bufferMilliseconds * (BitsPerSample / 8) * Channels / 1000;
            _bufferSize = Math.Max(_bufferSize, desiredBufferSize);

            System.Diagnostics.Debug.WriteLine($"AndroidAudioCaptureService 初始化完成，缓冲区大小: {_bufferSize}");
        }

        private async Task<bool> RequestMicrophonePermission()
        {
            try
            {
                var status = await Permissions.CheckStatusAsync<Permissions.Microphone>();

                if (status == PermissionStatus.Granted)
                    return true;

                if (Permissions.ShouldShowRationale<Permissions.Microphone>())
                {
                    // 可以在这里向用户解释为什么需要权限
                    System.Diagnostics.Debug.WriteLine("需要向用户说明麦克风权限用途");
                }

                status = await Permissions.RequestAsync<Permissions.Microphone>();
                return status == PermissionStatus.Granted;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"权限请求错误: {ex.Message}");
                return false;
            }
        }

        public override async Task StartCapture()
        {
            if (IsCapturing) return;

            lock (_disposeLock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(AndroidAudioCaptureService));

                try
                {
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        if (!await RequestMicrophonePermission())
                        {
                            throw new InvalidOperationException("麦克风权限被拒绝，无法录制音频");
                        }

                        await InitializeAudioRecord();
                    });
                    Task.Delay(2000).Wait(); // 等待初始化完成
                    IsCapturing = true;
                    _cancellationTokenSource = new CancellationTokenSource();

                    // 开始读取音频数据
                    _readingTask = Task.Run(() => ReadAudioData(_cancellationTokenSource.Token));

                    System.Diagnostics.Debug.WriteLine($"[{DateTime.Now:HH:mm:ss}] Android 麦克风实时采集已启动");
                }
                catch (AggregateException ex)
                {
                    throw ex.InnerException ?? ex;
                }
            }
        }

        private Task InitializeAudioRecord()
        {
            return Task.Run(() =>
            {
                lock (_disposeLock)
                {
                    if (_disposed) return;

                    try
                    {
                        _audioRecord = new AudioRecord(
                            AudioSource.Mic,
                            SampleRate,
                            ChannelConfig,
                            AudioFormat,
                            _bufferSize * 2); // 使用双倍缓冲区

                        // 检查 AudioRecord 状态
                        if (_audioRecord.State != State.Initialized)
                        {
                            throw new InvalidOperationException($"无法初始化 Android AudioRecord，状态: {_audioRecord.State}");
                        }

                        _audioRecord.StartRecording();

                        // 检查录音状态
                        if (_audioRecord.RecordingState != RecordState.Recording)
                        {
                            throw new InvalidOperationException($"AudioRecord 启动失败，录音状态: {_audioRecord.RecordingState}");
                        }

                        System.Diagnostics.Debug.WriteLine($"Android AudioRecord 已启动 - 采样率: {SampleRate}, 缓冲区: {_bufferSize}");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"初始化 AudioRecord 失败: {ex.Message}");
                        _audioRecord?.Release();
                        _audioRecord = null;
                        throw;
                    }
                }
            });
        }

        private async Task ReadAudioData(CancellationToken cancellationToken)
        {
            var buffer = new byte[_bufferSize];

            System.Diagnostics.Debug.WriteLine("开始读取音频数据循环");

            while (IsCapturing && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (_audioRecord == null || _audioRecord.RecordingState != RecordState.Recording)
                    {
                        await Task.Delay(10, cancellationToken);
                        continue;
                    }

                    // 使用 ReadAsync 方法读取音频数据
                    int bytesRead = await _audioRecord.ReadAsync(buffer, 0, _bufferSize);

                    if (bytesRead > 0 && !cancellationToken.IsCancellationRequested)
                    {
                        var audioData = new byte[bytesRead];
                        Buffer.BlockCopy(buffer, 0, audioData, 0, bytesRead);

                        // 转换为浮点数组并加入队列
                        float[] normalizedSamples = ConvertToFloatSamples(audioData, bytesRead);
                        _audioChunkQueue.Enqueue(normalizedSamples);

                        System.Diagnostics.Debug.WriteLine($"读取到 {bytesRead} 字节音频数据，转换为 {normalizedSamples.Length} 个采样点");
                    }
                    else if (bytesRead == (int)TrackStatus.ErrorInvalidOperation)
                    {
                        System.Diagnostics.Debug.WriteLine("AudioRecord 读取错误: ErrorInvalidOperation");
                        break;
                    }
                    else if (bytesRead == (int)TrackStatus.ErrorBadValue)
                    {
                        System.Diagnostics.Debug.WriteLine("AudioRecord 读取错误: ErrorBadValue");
                        break;
                    }
                    else if (bytesRead < 0)
                    {
                        System.Diagnostics.Debug.WriteLine($"AudioRecord 读取错误，错误代码: {bytesRead}");
                        break;
                    }

                    // 小延迟以避免过度占用 CPU
                    await Task.Delay(1, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    System.Diagnostics.Debug.WriteLine("音频数据读取操作被取消");
                    break;
                }
                catch (Java.Lang.IllegalStateException ex)
                {
                    System.Diagnostics.Debug.WriteLine($"AudioRecord 状态异常: {ex.Message}");
                    break;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"读取音频数据错误: {ex.Message}");
                    break;
                }
            }

            System.Diagnostics.Debug.WriteLine("Android 音频数据读取循环结束");
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
            if (!IsCapturing) return;

            lock (_disposeLock)
            {
                System.Diagnostics.Debug.WriteLine("开始停止 Android 音频采集");

                IsCapturing = false;
                _cancellationTokenSource?.Cancel();

                try
                {
                    // 等待读取任务完成（最多等待1秒）
                    if (_readingTask != null && !_readingTask.IsCompleted)
                    {
                        _readingTask.Wait(1000);
                    }
                }
                catch (AggregateException ex)
                {
                    System.Diagnostics.Debug.WriteLine($"停止读取任务时发生错误: {ex.InnerException?.Message}");
                }

                try
                {
                    if (_audioRecord != null)
                    {
                        if (_audioRecord.RecordingState == RecordState.Recording)
                        {
                            _audioRecord.Stop();
                            System.Diagnostics.Debug.WriteLine("AudioRecord 已停止");
                        }
                        _audioRecord.Release();
                        System.Diagnostics.Debug.WriteLine("AudioRecord 已释放");
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"停止 AudioRecord 错误: {ex.Message}");
                }
                finally
                {
                    _audioRecord = null;
                    _cancellationTokenSource?.Dispose();
                    _cancellationTokenSource = null;
                }

                // 发送结束信号
                _audioChunkQueue.Enqueue(null);

                System.Diagnostics.Debug.WriteLine($"[{DateTime.Now:HH:mm:ss}] Android 麦克风实时采集已停止");
            }
        }

        public override async Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken)
        {
            try
            {
                // 等待数据可用或取消
                int waitCount = 0;
                while (IsCapturing && _audioChunkQueue.IsEmpty)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        System.Diagnostics.Debug.WriteLine("GetNextMicChunkAsync: 操作被取消");
                        return null;
                    }

                    await Task.Delay(10, cancellationToken);
                    waitCount++;

                    // 每等待100次（约1秒）输出一次调试信息
                    if (waitCount % 100 == 0)
                    {
                        System.Diagnostics.Debug.WriteLine($"GetNextMicChunkAsync: 等待音频数据中... ({waitCount * 10}ms)");
                    }
                }

                if (!_audioChunkQueue.TryDequeue(out float[]? chunk) || cancellationToken.IsCancellationRequested)
                {
                    System.Diagnostics.Debug.WriteLine("GetNextMicChunkAsync: 无法获取数据或操作被取消");
                    return null;
                }

                // null 表示结束信号
                if (chunk == null)
                {
                    System.Diagnostics.Debug.WriteLine("GetNextMicChunkAsync: 收到结束信号");
                    return null;
                }

                System.Diagnostics.Debug.WriteLine($"GetNextMicChunkAsync: 获取到 {chunk.Length} 个采样点");
                return new List<List<float[]>> { new List<float[]> { chunk } };
            }
            catch (TaskCanceledException ex)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    System.Diagnostics.Debug.WriteLine("GetNextMicChunkAsync: 任务被正常取消，返回null");
                    return null;
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"GetNextMicChunkAsync: 意外取消 - {ex.Message}");
                    throw;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"GetNextMicChunkAsync: 其他异常 - {ex.Message}");
                return null;
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            lock (_disposeLock)
            {
                if (_disposed) return;

                System.Diagnostics.Debug.WriteLine("开始释放 AndroidAudioCaptureService 资源");

                if (disposing)
                {
                    // 停止采集
                    StopCapture();

                    // 清理托管资源
                    _cancellationTokenSource?.Dispose();

                    // 清空队列
                    while (_audioChunkQueue.TryDequeue(out _)) { }
                }

                // 释放非托管资源
                if (_audioRecord != null)
                {
                    try
                    {
                        if (_audioRecord.RecordingState == RecordState.Recording)
                        {
                            _audioRecord.Stop();
                        }
                        _audioRecord.Release();
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"释放 AudioRecord 资源时出错: {ex.Message}");
                    }
                    finally
                    {
                        _audioRecord = null;
                    }
                }

                _disposed = true;
                System.Diagnostics.Debug.WriteLine("AndroidAudioCaptureService 资源释放完成");
            }
        }

        public override void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~AndroidAudioCaptureService()
        {
            Dispose(false);
        }
    }
}