using AudioInOut.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace AudioInOut.Player
{
    internal class MacCoreAudioPlayer : BasePlayer, IDisposable
    {
        #region Core Audio Native Methods
        private static class CoreAudioNative
        {
            private const string CoreAudioLib = "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox";

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueNewOutput(ref AudioStreamBasicDescription inFormat,
                AudioQueueOutputCallback inCallbackProc, IntPtr inUserData, IntPtr inCallbackRunLoop,
                IntPtr inCallbackRunLoopMode, uint inFlags, out IntPtr outAQ);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueAllocateBuffer(IntPtr inAQ, uint inBufferByteSize, out IntPtr outBuffer);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueEnqueueBuffer(IntPtr inAQ, IntPtr inBuffer, uint inNumPacketDescs, IntPtr inPacketDescs);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueStart(IntPtr inAQ, IntPtr inStartTime);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueStop(IntPtr inAQ, bool inImmediate);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueuePause(IntPtr inAQ);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueDispose(IntPtr inAQ, bool inImmediate);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueFreeBuffer(IntPtr inAQ, IntPtr inBuffer);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueGetProperty(IntPtr inAQ, uint inID, IntPtr outData, ref uint ioDataSize);

            [DllImport(CoreAudioLib)]
            public static extern int AudioQueueSetProperty(IntPtr inAQ, uint inID, IntPtr inData, uint inDataSize);

            [DllImport(CoreAudioLib)]
            public static extern int AudioObjectGetPropertyData(uint inObjectID, ref AudioObjectPropertyAddress inAddress,
                uint inQualifierDataSize, IntPtr inQualifierData, ref uint ioDataSize, IntPtr outData);

            public delegate void AudioQueueOutputCallback(IntPtr inUserData, IntPtr inAQ, IntPtr inBuffer);

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

            [StructLayout(LayoutKind.Sequential)]
            public struct AudioObjectPropertyAddress
            {
                public uint mSelector;
                public uint mScope;
                public uint mElement;
            }

            // Constants
            public const uint kAudioFormatLinearPCM = 1819304813;
            public const uint kAudioFormatFlagIsFloat = 1;
            public const uint kAudioFormatFlagIsPacked = 8;
            public const uint kAudioQueueProperty_IsRunning = 1634825071;
            public const uint kAudioObjectPropertyElementMain = 0;
            public const uint kAudioObjectPropertyScopeGlobal = 1;
            public const uint kAudioObjectSystemObject = 1;
        }
        #endregion

        #region 字段
        private IntPtr _audioQueue = IntPtr.Zero;
        private List<IntPtr> _audioBuffers = new List<IntPtr>();
        private CoreAudioNative.AudioQueueOutputCallback? _callback;
        private int _currentSampleRate = 44100;
        private int _currentChannels = 2;
        private bool _isPaused = false;
        private bool _isRunning = false;
        private readonly object _lockObject = new object();
        #endregion

        #region 属性实现
        public override string Name => "macOS Core Audio Player";

        public override AudioPlaybackState State
        {
            get
            {
                if (_audioQueue == IntPtr.Zero)
                    return AudioPlaybackState.Stopped;

                if (_isPaused)
                    return AudioPlaybackState.Paused;

                return _isRunning ? AudioPlaybackState.Playing : AudioPlaybackState.Stopped;
            }
        }

        public override bool IsActivated => _audioQueue != IntPtr.Zero && _isRunning;
        #endregion

        #region 设备管理实现
        public override IReadOnlyList<AudioDeviceInfo> GetAudioDevices()
        {
            var devices = new List<AudioDeviceInfo>();

            try
            {
                // 对于Core Audio，我们通常使用默认输出设备
                // 更复杂的设备枚举需要额外的AudioHardwareService API调用
                devices.Add(new AudioDeviceInfo
                {
                    Id = "default",
                    Name = "Default Output Device",
                    Channels = 2,
                    SupportedSampleRates = new[] { 8000, 16000, 22050, 24000, 44100, 48000, 96000 },
                    IsDefault = true
                });

                // 这里可以添加更多设备枚举逻辑
                // 需要调用AudioHardwareService API来获取设备列表
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"Failed to enumerate Core Audio devices: {ex.Message}"));
            }

            return devices;
        }

        public override bool SelectDevice(string deviceId)
        {
            // Core Audio 通常使用默认设备，设备选择需要重新初始化
            var device = GetAudioDevices().FirstOrDefault(d => d.Id == deviceId);
            if (device != null)
            {
                _currentDevice = device;
                return true;
            }
            return false;
        }
        #endregion

        #region 抽象方法实现
        protected override bool InitializeDevice()
        {
            lock (_lockObject)
            {
                if (_audioQueue != IntPtr.Zero)
                {
                    CloseDevice();
                }

                try
                {
                    // 创建回调委托实例（防止被GC回收）
                    _callback = new CoreAudioNative.AudioQueueOutputCallback(AudioQueueOutputCallback);

                    var format = new CoreAudioNative.AudioStreamBasicDescription
                    {
                        mSampleRate = _currentSampleRate, // 默认采样率，会在播放时调整
                        mFormatID = CoreAudioNative.kAudioFormatLinearPCM,
                        mFormatFlags = CoreAudioNative.kAudioFormatFlagIsFloat | CoreAudioNative.kAudioFormatFlagIsPacked,
                        mBytesPerPacket = (uint)_currentChannels * 4, // 默认2通道×4字节
                        mFramesPerPacket = 1,
                        mBytesPerFrame = (uint)_currentChannels * 4, // 声道数 × 4字节(32位浮点)
                        mChannelsPerFrame = (uint)_currentChannels, // 默认立体声
                        mBitsPerChannel = 32,
                        mReserved = 0
                    };

                    int result = CoreAudioNative.AudioQueueNewOutput(ref format,
                        _callback, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 0, out _audioQueue);

                    if (result != 0)
                    {
                        OnErrorOccurred(new Exception($"Core Audio initialization failed with error: {result}"));
                        return false;
                    }

                    _isRunning = false;
                    _isPaused = false;
                    return true;
                }
                catch (Exception ex)
                {
                    OnErrorOccurred(new Exception($"Core Audio initialization error: {ex.Message}"));
                    return false;
                }
            }
        }

        protected override bool PlaySampleInternal(SampleEntity sample)
        {
            if (_audioQueue == IntPtr.Zero || sample.Sample == null || sample.Sample.Length == 0)
                return false;

            lock (_lockObject)
            {
                try
                {
                    // 检查是否需要重新配置格式
                    if (sample.SampleRate != _currentSampleRate || sample.Channels != _currentChannels)
                    {
                        if (!ConfigureAudioFormat(sample.SampleRate, sample.Channels))
                            return false;
                    }

                    uint bufferSize = (uint)(sample.Sample.Length * 4); // 4 bytes per float
                    int result = CoreAudioNative.AudioQueueAllocateBuffer(_audioQueue, bufferSize, out IntPtr buffer);

                    if (result != 0)
                    {
                        OnErrorOccurred(new Exception($"Failed to allocate audio buffer: {result}"), sample);
                        return false;
                    }

                    // 复制样本数据到缓冲区
                    unsafe
                    {
                        float* bufferPtr = (float*)buffer;
                        for (int i = 0; i < sample.Sample.Length; i++)
                        {
                            bufferPtr[i] = sample.Sample[i];
                        }
                    }

                    result = CoreAudioNative.AudioQueueEnqueueBuffer(_audioQueue, buffer, 0, IntPtr.Zero);
                    if (result != 0)
                    {
                        CoreAudioNative.AudioQueueFreeBuffer(_audioQueue, buffer);
                        OnErrorOccurred(new Exception($"Failed to enqueue audio buffer: {result}"), sample);
                        return false;
                    }

                    _audioBuffers.Add(buffer);

                    // 如果没有运行，开始播放
                    if (!_isRunning && !_isPaused)
                    {
                        result = CoreAudioNative.AudioQueueStart(_audioQueue, IntPtr.Zero);
                        if (result == 0)
                        {
                            _isRunning = true;
                        }
                    }

                    return result == 0;
                }
                catch (Exception ex)
                {
                    OnErrorOccurred(new Exception($"Core Audio playback error: {ex.Message}"), sample);
                    return false;
                }
            }
        }

        protected override void CloseDevice()
        {
            lock (_lockObject)
            {
                if (_audioQueue != IntPtr.Zero)
                {
                    try
                    {
                        CoreAudioNative.AudioQueueStop(_audioQueue, true);

                        // 释放所有缓冲区
                        foreach (var buffer in _audioBuffers)
                        {
                            CoreAudioNative.AudioQueueFreeBuffer(_audioQueue, buffer);
                        }
                        _audioBuffers.Clear();

                        CoreAudioNative.AudioQueueDispose(_audioQueue, true);
                    }
                    catch (Exception ex)
                    {
                        OnErrorOccurred(new Exception($"Core Audio close error: {ex.Message}"));
                    }
                    finally
                    {
                        _audioQueue = IntPtr.Zero;
                        _isRunning = false;
                        _isPaused = false;
                        _currentSampleRate = 0;
                        _currentChannels = 0;
                    }
                }
            }
        }

        protected override void PauseInternal()
        {
            lock (_lockObject)
            {
                if (_audioQueue != IntPtr.Zero && _isRunning && !_isPaused)
                {
                    int result = CoreAudioNative.AudioQueuePause(_audioQueue);
                    if (result == 0)
                    {
                        _isPaused = true;
                    }
                }
            }
        }

        protected override void ResumeInternal()
        {
            lock (_lockObject)
            {
                if (_audioQueue != IntPtr.Zero && _isPaused)
                {
                    int result = CoreAudioNative.AudioQueueStart(_audioQueue, IntPtr.Zero);
                    if (result == 0)
                    {
                        _isPaused = false;
                    }
                }
            }
        }
        #endregion

        #region 状态检查实现
        public override bool IsDeviceReady()
        {
            return _audioQueue != IntPtr.Zero;
        }

        public override bool IsFormatSupported(int sampleRate, int channels)
        {
            // Core Audio 通常支持常见的音频格式
            int[] supportedRates = { 8000, 11025, 16000, 22050, 24000, 44100, 48000, 96000 };
            return supportedRates.Contains(sampleRate) && channels >= 1 && channels <= 2;
        }

        public override int GetDeviceLatency()
        {
            // Core Audio 通常有较低的延迟
            return 20; // 估计20ms延迟
        }
        #endregion

        #region 回调方法和辅助方法
        private void AudioQueueOutputCallback(IntPtr inUserData, IntPtr inAQ, IntPtr inBuffer)
        {
            lock (_lockObject)
            {
                // 缓冲区播放完成，释放资源
                if (_audioBuffers.Contains(inBuffer))
                {
                    CoreAudioNative.AudioQueueFreeBuffer(inAQ, inBuffer);
                    _audioBuffers.Remove(inBuffer);
                }

                // 检查是否所有缓冲区都播放完成
                if (_audioBuffers.Count == 0 && _isRunning)
                {
                    _isRunning = false;
                }
            }
        }

        private bool ConfigureAudioFormat(int sampleRate, int channels)
        {
            if (_audioQueue == IntPtr.Zero)
                return false;

            // 对于Core Audio，我们需要重新创建AudioQueue来改变格式
            // 先关闭当前设备
            CloseDevice();

            // 重新初始化设备
            if (!InitializeDevice())
                return false;

            _currentSampleRate = sampleRate;
            _currentChannels = channels;
            return true;
        }
        #endregion

        #region IDisposable实现
        private bool _disposed = false;

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // 释放托管资源
                }

                // 释放非托管资源
                CloseDevice();
                _disposed = true;
            }

            base.Dispose(disposing);
        }
        #endregion
    }
}