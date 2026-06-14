using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AudioInOut.Base;

namespace AudioInOut.Recorder
{
    // 回调委托
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void AudioQueueInputCallback(nint inUserData, nint inAQ, nint inBuffer,
        nint inStartTime, ulong inNumPackets, nint inPacketDesc);

    internal class MacCoreAudioRecorder : BaseRecorder, IDisposable
    {
        private nint _audioQueue = default(nint);
        private bool _isCapturing = false;
        private readonly ConcurrentQueue<float[]> _audioChunkQueue;
        private readonly int _bufferMilliseconds;

        public bool IsCapturing => _isCapturing;
        public const int SampleRate = 16000;
        public const int BitsPerSample = 16;
        public const int Channels = 1;

        private bool _isPaused = false;
        private readonly object _pauseLock = new object();
        private List<nint> _bufferList = new List<nint>(); // 保存缓冲区列表
        /// <summary>
        /// 是否已暂停
        /// </summary>
        public bool IsPaused => _isPaused;

        public MacCoreAudioRecorder(int bufferMilliseconds = 100)
        {
            _bufferMilliseconds = bufferMilliseconds;
            _audioChunkQueue = new ConcurrentQueue<float[]>();
        }

        public override async Task StartCapture()
        {
            if (_isCapturing) return;

            try
            {
                // 创建音频格式描述
                var format = new CoreAudioNative.AudioStreamBasicDescription
                {
                    mSampleRate = SampleRate,
                    mFormatID = CoreAudioNative.kAudioFormatLinearPCM,
                    mFormatFlags = CoreAudioNative.kAudioFormatFlagIsSignedInteger | CoreAudioNative.kAudioFormatFlagIsPacked,
                    mBytesPerPacket = (uint)(BitsPerSample / 8 * Channels),
                    mFramesPerPacket = 1,
                    mBytesPerFrame = (uint)(BitsPerSample / 8 * Channels),
                    mChannelsPerFrame = (uint)Channels,
                    mBitsPerChannel = (uint)BitsPerSample,
                    mReserved = 0
                };

                // 创建音频队列
                AudioQueueInputCallback callback = AudioInputCallback;
                GCHandle callbackHandle = GCHandle.Alloc(callback);

                int result = CoreAudioNative.AudioQueueNewInput(
                    ref format,
                    callback,
                    default(nint),
                    default(nint),
                    default(nint),
                    0,
                    out _audioQueue);

                if (result != 0)
                {
                    throw new InvalidOperationException($"CoreAudio create failed: {result}");
                }

                // 分配和启用缓冲区
                SetupBuffers();

                // 开始录制
                result = CoreAudioNative.AudioQueueStart(_audioQueue, default(nint));
                if (result != 0)
                {
                    throw new InvalidOperationException($"CoreAudio start failed: {result}");
                }

                _isCapturing = true;
                _isPaused = false;
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] macOS CoreAudio 麦克风采集已启动");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CoreAudio start failed: {ex.Message}");
                throw;
            }
        }

        private void SetupBuffers()
        {
            int bufferSize = SampleRate * _bufferMilliseconds * (BitsPerSample / 8) * Channels / 1000;
            const int bufferCount = 3;
            _bufferList.Clear();

            for (int i = 0; i < bufferCount; i++)
            {
                nint bufferPtr;
                int result = CoreAudioNative.AudioQueueAllocateBuffer(_audioQueue, (uint)bufferSize, out bufferPtr);
                if (result != 0)
                {
                    throw new InvalidOperationException($"CoreAudio buffer allocation failed: {result}");
                }

                result = CoreAudioNative.AudioQueueEnqueueBuffer(_audioQueue, bufferPtr, 0, default(nint));
                if (result != 0)
                {
                    throw new InvalidOperationException($"CoreAudio buffer enqueue failed: {result}");
                }

                _bufferList.Add(bufferPtr); // 保存缓冲区指针
            }
        }

        private void AudioInputCallback(nint inUserData, nint inAQ, nint inBuffer, nint inStartTime, ulong inNumPackets, nint inPacketDesc)
        {
            // 如果处于暂停状态，仍然重新提交缓冲区但不处理数据
            if (!_isCapturing) return;

            try
            {
                if (!_isPaused)
                {
                    // 获取音频数据
                    nint audioDataPtr = CoreAudioNative.AudioQueueBufferGetAudioData(inBuffer);
                    uint dataByteSize = CoreAudioNative.AudioQueueBufferGetAudioDataByteSize(inBuffer);

                    if (dataByteSize > 0)
                    {
                        byte[] audioData = new byte[dataByteSize];
                        Marshal.Copy(audioDataPtr, audioData, 0, (int)dataByteSize);

                        float[] normalizedSamples = ConvertToFloatSamples(audioData, (int)dataByteSize);
                        _audioChunkQueue.Enqueue(normalizedSamples);
                    }
                }

                // 始终重新提交缓冲区以保持队列运行
                CoreAudioNative.AudioQueueEnqueueBuffer(_audioQueue, inBuffer, 0, default(nint));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CoreAudio callback error: {ex.Message}");
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
            _isPaused = false;

            if (_audioQueue != default(nint))
            {
                CoreAudioNative.AudioQueueStop(_audioQueue, true);
                CoreAudioNative.AudioQueueDispose(_audioQueue, true);
                _audioQueue = default(nint);
            }

            _bufferList.Clear();
            _audioChunkQueue.Enqueue(null);
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] macOS CoreAudio 麦克风采集已停止");
        }

        public override void PauseCapture()
        {
            lock (_pauseLock)
            {
                if (!_isCapturing || _isPaused) return;

                if (_audioQueue != default(nint))
                {
                    // 使用 AudioQueuePause 如果可用，或者使用带缓冲区的停止
                    // 注意：标准 AudioQueue API 没有直接提供 Pause 方法

                    // 方法：标记为暂停，在回调中停止处理数据
                    _isPaused = true;

                    // 可选：临时停止 AudioQueue 但不释放资源
                    int result = CoreAudioNative.AudioQueueStop(_audioQueue, false);
                    if (result != 0)
                    {
                        Console.WriteLine($"CoreAudio pause stop failed: {result}");
                    }

                    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] macOS CoreAudio 麦克风采集已暂停");
                }
            }
        }

        public override void ResumeCapture()
        {
            lock (_pauseLock)
            {
                if (!_isCapturing || !_isPaused) return;

                if (_audioQueue != default(nint))
                {
                    // 如果之前调用了 AudioQueueStop(false)，现在只需要重新启动
                    int result = CoreAudioNative.AudioQueueStart(_audioQueue, default(nint));
                    if (result != 0)
                    {
                        throw new InvalidOperationException($"CoreAudio resume failed: {result}");
                    }

                    _isPaused = false;
                    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] macOS CoreAudio 麦克风采集已恢复");
                }
            }
        }

        public override async Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken)
        {
            try
            {
                // 等待时有数据或暂停时等待恢复
                while ((_isCapturing && _audioChunkQueue.IsEmpty) || (_isCapturing && _isPaused))
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
            catch (TaskCanceledException)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    Console.WriteLine("The task of get mic data: cancelled normally, return null");
                    return null;
                }
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"The task of get mic data: other exceptions - {ex.Message}");
                return null;
            }
        }

        //public void Dispose()
        //{
        //    StopCapture();
        //}
    }

    // Core Audio Native 互操作
    internal static class CoreAudioNative
    {
        public const uint kAudioFormatLinearPCM = 0x6C70636D; // 'lpcm'
        public const uint kAudioFormatFlagIsSignedInteger = 1 << 0;
        public const uint kAudioFormatFlagIsPacked = 1 << 3;

        [StructLayout(LayoutKind.Sequential)]
        public struct AudioStreamBasicDescription
        {
            public double mSampleRate;
            public uint mFormatID;
            public uint mFormatFlags;
            public uint mBytesPerPacket;
            public uint mFramesPerPacket;
            public uint mBytesPerFrame;
            public uint mChannelsPerFrame;
            public uint mBitsPerChannel;
            public uint mReserved;
        }

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueNewInput(
            ref AudioStreamBasicDescription inFormat,
            AudioQueueInputCallback inCallback,
            nint inUserData,
            nint inCallbackRunLoop,
            nint inCallbackRunLoopMode,
            uint inFlags,
            out nint outAQ);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueStart(nint inAQ, nint inStartTime);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueStop(nint inAQ, bool inImmediate);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueDispose(nint inAQ, bool inImmediate);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueAllocateBuffer(nint inAQ, uint inBufferByteSize, out nint outBuffer);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern int AudioQueueEnqueueBuffer(nint inAQ, nint inBuffer, uint inNumPacketDescs, nint inPacketDescs);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern nint AudioQueueBufferGetAudioData(nint inBuffer);

        [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
        public static extern uint AudioQueueBufferGetAudioDataByteSize(nint inBuffer);
    }
}