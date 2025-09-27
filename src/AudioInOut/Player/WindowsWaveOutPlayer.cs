using AudioInOut.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace AudioInOut.Player
{
    internal class WindowsWaveOutPlayer : BasePlayer, IDisposable
    {
        #region WaveOut Native Methods
        private static class WaveOutNative
        {
            private const string WinMM = "winmm.dll";

            [DllImport(WinMM)]
            public static extern int waveOutOpen(out IntPtr hWaveOut, int uDeviceID,
                ref WAVEFORMATEX lpFormat, WaveOutProc dwCallback, IntPtr dwInstance, int fdwOpen);

            [DllImport(WinMM)]
            public static extern int waveOutPrepareHeader(IntPtr hWaveOut, ref WAVEHDR lpWaveOutHdr, int uSize);

            [DllImport(WinMM)]
            public static extern int waveOutWrite(IntPtr hWaveOut, ref WAVEHDR lpWaveOutHdr, int uSize);

            [DllImport(WinMM)]
            public static extern int waveOutUnprepareHeader(IntPtr hWaveOut, ref WAVEHDR lpWaveOutHdr, int uSize);

            [DllImport(WinMM)]
            public static extern int waveOutClose(IntPtr hWaveOut);

            [DllImport(WinMM)]
            public static extern int waveOutReset(IntPtr hWaveOut);

            [DllImport(WinMM)]
            public static extern int waveOutPause(IntPtr hWaveOut);

            [DllImport(WinMM)]
            public static extern int waveOutRestart(IntPtr hWaveOut);

            [DllImport(WinMM)]
            public static extern int waveOutGetVolume(IntPtr hWaveOut, out uint dwVolume);

            [DllImport(WinMM)]
            public static extern int waveOutSetVolume(IntPtr hWaveOut, uint dwVolume);

            [DllImport(WinMM)]
            public static extern int waveOutGetNumDevs();

            [DllImport(WinMM)]
            public static extern int waveOutGetDevCaps(IntPtr uDeviceID, out WAVEOUTCAPS pwoc, int cbwoc);

            [DllImport(WinMM)]
            public static extern int waveOutGetErrorText(int mmrError, StringBuilder pszText, int cchText);

            public delegate void WaveOutProc(IntPtr hWaveOut, int uMsg, IntPtr dwInstance, IntPtr dwParam1, IntPtr dwParam2);

            [StructLayout(LayoutKind.Sequential)]
            public struct WAVEFORMATEX
            {
                public ushort wFormatTag;
                public ushort nChannels;
                public uint nSamplesPerSec;
                public uint nAvgBytesPerSec;
                public ushort nBlockAlign;
                public ushort wBitsPerSample;
                public ushort cbSize;
            }

            [StructLayout(LayoutKind.Sequential)]
            public struct WAVEHDR
            {
                public IntPtr lpData;
                public uint dwBufferLength;
                public uint dwBytesRecorded;
                public IntPtr dwUser;
                public uint dwFlags;
                public uint dwLoops;
                public IntPtr lpNext;
                public IntPtr reserved;
            }

            [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
            public struct WAVEOUTCAPS
            {
                public ushort wMid;
                public ushort wPid;
                public uint vDriverVersion;
                [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
                public string szPname;
                public uint dwFormats;
                public ushort wChannels;
                public ushort wReserved1;
                public uint dwSupport;
            }

            // Constants
            public const int WAVE_MAPPER = -1;
            public const uint WAVE_FORMAT_IEEE_FLOAT = 3;
            public const int CALLBACK_FUNCTION = 0x30000;
            public const int MM_WOM_OPEN = 0x3BB;
            public const int MM_WOM_CLOSE = 0x3BC;
            public const int MM_WOM_DONE = 0x3BD;
            public const uint WHDR_DONE = 0x00000001;
        }
        #endregion

        #region 字段
        private IntPtr _waveOutHandle = IntPtr.Zero;
        private readonly List<WaveOutBuffer> _activeBuffers = new List<WaveOutBuffer>();
        private int _currentSampleRate = 44100;
        private int _currentChannels = 2;
        private bool _isPaused = false;
        private bool _isPlaying = false;
        private readonly object _lockObject = new object();
        private WaveOutNative.WaveOutProc? _callback;
        #endregion

        #region 内部缓冲区类
        private class WaveOutBuffer : IDisposable
        {
            public IntPtr HeaderPointer { get; }
            public IntPtr DataPointer { get; }
            public int Size { get; }
            public bool IsPrepared { get; set; }
            public SampleEntity? Sample { get; set; }

            public WaveOutBuffer(int size)
            {
                Size = size;
                DataPointer = Marshal.AllocHGlobal(size);

                var header = new WaveOutNative.WAVEHDR
                {
                    lpData = DataPointer,
                    dwBufferLength = (uint)size,
                    dwFlags = 0,
                    dwLoops = 0
                };

                HeaderPointer = Marshal.AllocHGlobal(Marshal.SizeOf(header));
                Marshal.StructureToPtr(header, HeaderPointer, false);
            }

            public void Dispose()
            {
                if (HeaderPointer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(HeaderPointer);
                }
                if (DataPointer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(DataPointer);
                }
            }
        }
        #endregion

        #region 属性实现
        public override string Name => "Windows WaveOut Player";

        public override AudioPlaybackState State
        {
            get
            {
                if (_waveOutHandle == IntPtr.Zero)
                    return AudioPlaybackState.Stopped;

                if (_isPaused)
                    return AudioPlaybackState.Paused;

                return _isPlaying ? AudioPlaybackState.Playing : AudioPlaybackState.Stopped;
            }
        }

        public override bool IsActivated => _waveOutHandle != IntPtr.Zero && _isPlaying;
        #endregion

        #region 设备管理实现
        public override IReadOnlyList<AudioDeviceInfo> GetAudioDevices()
        {
            var devices = new List<AudioDeviceInfo>();

            try
            {
                int deviceCount = WaveOutNative.waveOutGetNumDevs();

                for (int deviceId = 0; deviceId < deviceCount; deviceId++)
                {
                    int result = WaveOutNative.waveOutGetDevCaps((IntPtr)deviceId,
                        out WaveOutNative.WAVEOUTCAPS caps, Marshal.SizeOf<WaveOutNative.WAVEOUTCAPS>());

                    if (result == 0) // MMSYSERR_NOERROR
                    {
                        devices.Add(new AudioDeviceInfo
                        {
                            Id = deviceId.ToString(),
                            Name = caps.szPname,
                            Channels = caps.wChannels,
                            SupportedSampleRates = GetSupportedSampleRates(caps.dwFormats),
                            IsDefault = deviceId == WaveOutNative.WAVE_MAPPER
                        });
                    }
                }

                // 添加波形映射器作为默认设备
                if (devices.Count > 0)
                {
                    devices.Add(new AudioDeviceInfo
                    {
                        Id = WaveOutNative.WAVE_MAPPER.ToString(),
                        Name = "Wave Mapper (Default)",
                        Channels = 2,
                        SupportedSampleRates = new[] { 8000, 11025, 16000, 22050, 24000, 44100, 48000 },
                        IsDefault = true
                    });
                }
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"Failed to enumerate WaveOut devices: {ex.Message}"));
            }

            return devices;
        }

        public override bool SelectDevice(string deviceId)
        {
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
                if (_waveOutHandle != IntPtr.Zero)
                {
                    CloseDevice();
                }

                try
                {
                    // 创建回调委托实例（防止被GC回收）
                    _callback = new WaveOutNative.WaveOutProc(WaveOutCallback);
                    var format = new WaveOutNative.WAVEFORMATEX
                    {
                        wFormatTag = (ushort)WaveOutNative.WAVE_FORMAT_IEEE_FLOAT,
                        nChannels = (ushort)_currentChannels,
                        nSamplesPerSec = (uint)_currentSampleRate,
                        wBitsPerSample = 32,
                        nBlockAlign = (ushort)(_currentChannels * 4), // 1声道 × 4字节(32位)
                        nAvgBytesPerSec = (uint)(_currentSampleRate * _currentChannels * 4), // 采样率 × 声道 × 4字节(32位)
                        cbSize = 0
                    };

                    int deviceId = _currentDevice != null ? int.Parse(_currentDevice.Id) : WaveOutNative.WAVE_MAPPER;

                    int result = WaveOutNative.waveOutOpen(out _waveOutHandle, deviceId, ref format,
                        _callback, IntPtr.Zero, WaveOutNative.CALLBACK_FUNCTION);

                    if (result != 0)
                    {
                        string errorText = GetWaveOutErrorText(result);
                        OnErrorOccurred(new Exception($"WaveOut initialization failed: {errorText}"));
                        return false;
                    }

                    _isPlaying = false;
                    _isPaused = false;
                    return true;
                }
                catch (Exception ex)
                {
                    OnErrorOccurred(new Exception($"WaveOut initialization error: {ex.Message}"));
                    return false;
                }
            }
        }

        protected override bool PlaySampleInternal(SampleEntity sample)
        {
            if (_waveOutHandle == IntPtr.Zero || sample.Sample == null || sample.Sample.Length == 0)
                return false;

            lock (_lockObject)
            {
                try
                {
                    // 检查是否需要重新配置格式
                    if (sample.SampleRate != _currentSampleRate || sample.Channels != _currentChannels)
                    {
                        if (!ReconfigureWaveOut(sample.SampleRate, sample.Channels))
                            return false;
                    }

                    int bufferSize = sample.Sample.Length * 4; // 4 bytes per float
                    var buffer = new WaveOutBuffer(bufferSize)
                    {
                        Sample = sample
                    };

                    // 复制样本数据到缓冲区
                    Marshal.Copy(sample.Sample, 0, buffer.DataPointer, sample.Sample.Length);

                    // 准备缓冲区头
                    var waveHeader = GetHeaderFromPointer(buffer.HeaderPointer);
                    //waveHeader.lpData = buffer.DataPointer;
                    //waveHeader.dwBufferLength = (uint)sample.Sample.Length * 4;
                    //waveHeader.dwFlags = 1;

                    int result = WaveOutNative.waveOutPrepareHeader(_waveOutHandle,
                        ref waveHeader, Marshal.SizeOf<WaveOutNative.WAVEHDR>());

                    if (result != 0)
                    {
                        buffer.Dispose();
                        OnErrorOccurred(new Exception($"Failed to prepare wave header: {GetWaveOutErrorText(result)}"), sample);
                        return false;
                    }

                    buffer.IsPrepared = true;

                    // 写入缓冲区
                    result = WaveOutNative.waveOutWrite(_waveOutHandle,
                        ref waveHeader, Marshal.SizeOf<WaveOutNative.WAVEHDR>());

                    // 等待播放完成
                    while ((waveHeader.dwFlags & WaveOutNative.WHDR_DONE) == 0)
                    {
                        Thread.Sleep(10);
                    }

                    if (result != 0)
                    {
                        CleanupBuffer(buffer);
                        OnErrorOccurred(new Exception($"Failed to write wave data: {GetWaveOutErrorText(result)}"), sample);
                        return false;
                    }

                    _activeBuffers.Add(buffer);
                    _isPlaying = true;

                    return true;
                }
                catch (Exception ex)
                {
                    OnErrorOccurred(new Exception($"WaveOut playback error: {ex.Message}"), sample);
                    return false;
                }
            }
        }

        protected override void CloseDevice()
        {
            lock (_lockObject)
            {
                if (_waveOutHandle != IntPtr.Zero)
                {
                    try
                    {
                        // 重置设备，停止所有播放
                        WaveOutNative.waveOutReset(_waveOutHandle);

                        // 清理所有活动缓冲区
                        foreach (var buffer in _activeBuffers.ToArray())
                        {
                            CleanupBuffer(buffer);
                        }
                        _activeBuffers.Clear();

                        WaveOutNative.waveOutClose(_waveOutHandle);
                    }
                    catch (Exception ex)
                    {
                        OnErrorOccurred(new Exception($"WaveOut close error: {ex.Message}"));
                    }
                    finally
                    {
                        _waveOutHandle = IntPtr.Zero;
                        _isPlaying = false;
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
                if (_waveOutHandle != IntPtr.Zero && _isPlaying && !_isPaused)
                {
                    int result = WaveOutNative.waveOutPause(_waveOutHandle);
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
                if (_waveOutHandle != IntPtr.Zero && _isPaused)
                {
                    int result = WaveOutNative.waveOutRestart(_waveOutHandle);
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
            return _waveOutHandle != IntPtr.Zero;
        }

        public override bool IsFormatSupported(int sampleRate, int channels)
        {
            // WaveOut 通常支持常见的音频格式
            int[] supportedRates = { 8000, 11025, 16000, 22050, 24000, 44100, 48000 };
            return supportedRates.Contains(sampleRate) && channels >= 1 && channels <= 2;
        }

        public override int GetDeviceLatency()
        {
            // WaveOut 通常有较高的延迟
            return 100; // 估计100ms延迟
        }
        #endregion

        #region 辅助方法
        private void WaveOutCallback(IntPtr hWaveOut, int uMsg, IntPtr dwInstance, IntPtr dwParam1, IntPtr dwParam2)
        {
            if (uMsg == WaveOutNative.MM_WOM_DONE)
            {
                lock (_lockObject)
                {
                    var buffer = _activeBuffers.FirstOrDefault(b => b.HeaderPointer == dwParam1);
                    if (buffer != null)
                    {
                        CleanupBuffer(buffer);
                        _activeBuffers.Remove(buffer);

                        // 如果没有活动缓冲区，播放完成
                        if (_activeBuffers.Count == 0)
                        {
                            _isPlaying = false;
                        }
                    }
                }
            }
        }

        private void CleanupBuffer(WaveOutBuffer buffer)
        {
            if (buffer.IsPrepared && _waveOutHandle != IntPtr.Zero)
            {
                try
                {
                    var header = GetHeaderFromPointer(buffer.HeaderPointer);
                    WaveOutNative.waveOutUnprepareHeader(_waveOutHandle,
                        ref header,
                        Marshal.SizeOf<WaveOutNative.WAVEHDR>());
                }
                catch
                {
                    // 忽略清理错误
                }
            }
            buffer.Dispose();
        }

        private bool ReconfigureWaveOut(int sampleRate, int channels)
        {
            // 关闭当前设备并重新初始化
            CloseDevice();

            _currentSampleRate = sampleRate;
            _currentChannels = channels;

            return InitializeDevice();
        }

        private static int[] GetSupportedSampleRates(uint formats)
        {
            var rates = new List<int>();

            // 检查支持的采样率（简化实现）
            if ((formats & 0x00000001) != 0) rates.Add(11025); // WAVE_FORMAT_1M08
            if ((formats & 0x00000002) != 0) rates.Add(22050); // WAVE_FORMAT_1S08
            if ((formats & 0x00000004) != 0) rates.Add(24000);
            if ((formats & 0x00000004) != 0) rates.Add(44100); // WAVE_FORMAT_2S08
            if ((formats & 0x00000008) != 0) rates.Add(48000); // WAVE_FORMAT_4S08

            return rates.Count > 0 ? rates.ToArray() : new[] { 8000, 11025, 22050, 24000, 44100, 48000 };
        }

        private static string GetWaveOutErrorText(int errorCode)
        {
            var sb = new StringBuilder(256);
            WaveOutNative.waveOutGetErrorText(errorCode, sb, sb.Capacity);
            return sb.ToString();
        }

        private static WaveOutNative.WAVEHDR GetHeaderFromPointer(IntPtr headerPointer)
        {
            return Marshal.PtrToStructure<WaveOutNative.WAVEHDR>(headerPointer);
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