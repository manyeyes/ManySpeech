using AudioInOut.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace AudioInOut.Player
{
    public class LinuxAlsaPlayer : BasePlayer, IDisposable
    {
        #region ALSA Native Methods
        private static class AlsaNative
        {
            private const string AlsaLib = "libasound.so.2";

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_open(out IntPtr pcm, string name, int stream, int mode);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_set_params(IntPtr pcm, int format, int access, int channels,
                                                       int rate, int soft_resample, int latency);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_writei(IntPtr pcm, IntPtr buffer, int size);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_recover(IntPtr pcm, int result, int size);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_drain(IntPtr pcm);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_close(IntPtr pcm);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_prepare(IntPtr pcm);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_pause(IntPtr pcm, int enable);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_resume(IntPtr pcm);

            [DllImport(AlsaLib)]
            public static extern int snd_pcm_state(IntPtr pcm);

            [DllImport(AlsaLib)]
            public static extern IntPtr snd_strerror(int errnum);

            [DllImport(AlsaLib)]
            public static extern int snd_device_name_hint(int card, string iface, out IntPtr hints);

            [DllImport(AlsaLib)]
            public static extern IntPtr snd_device_name_get_hint(IntPtr hint, string id);

            [DllImport(AlsaLib)]
            public static extern int snd_device_name_free_hint(IntPtr hints);

            public const int SND_PCM_STREAM_PLAYBACK = 0;
            public const int SND_PCM_ACCESS_RW_INTERLEAVED = 3;
            public const int SND_PCM_FORMAT_FLOAT_LE = 14;
            public const int SND_PCM_STATE_RUNNING = 3;
            public const int SND_PCM_STATE_PAUSED = 4;
        }
        #endregion

        #region 字段
        private IntPtr _alsaHandle = IntPtr.Zero;
        private int _currentSampleRate = 44100;
        private int _currentChannels = 2;
        private bool _isPaused = false;
        #endregion

        #region 属性实现
        public override string Name => "ALSA Audio Player";

        public override AudioPlaybackState State
        {
            get
            {
                if (_alsaHandle == IntPtr.Zero)
                    return AudioPlaybackState.Stopped;

                if (_isPaused)
                    return AudioPlaybackState.Paused;

                int state = AlsaNative.snd_pcm_state(_alsaHandle);
                return state == AlsaNative.SND_PCM_STATE_RUNNING ?
                    AudioPlaybackState.Playing : AudioPlaybackState.Stopped;
            }
        }

        public override bool IsActivated => _alsaHandle != IntPtr.Zero;
        #endregion

        #region 设备管理实现
        public override IReadOnlyList<AudioDeviceInfo> GetAudioDevices()
        {
            var devices = new List<AudioDeviceInfo>();

            try
            {
                int result = AlsaNative.snd_device_name_hint(-1, "pcm", out IntPtr hints);
                if (result < 0)
                    return devices;

                IntPtr currentHint = hints;
                while (true)
                {
                    IntPtr namePtr = AlsaNative.snd_device_name_get_hint(currentHint, "NAME");
                    IntPtr descPtr = AlsaNative.snd_device_name_get_hint(currentHint, "DESC");
                    IntPtr ioidPtr = AlsaNative.snd_device_name_get_hint(currentHint, "IOID");

                    if (namePtr == IntPtr.Zero)
                        break;

                    string name = Marshal.PtrToStringAnsi(namePtr) ?? string.Empty;
                    string description = Marshal.PtrToStringAnsi(descPtr) ?? name;
                    string ioid = Marshal.PtrToStringAnsi(ioidPtr) ?? "Output";

                    // 只包括输出设备
                    if (ioid == "Output" || ioid == "Duplex" || string.IsNullOrEmpty(ioid))
                    {
                        devices.Add(new AudioDeviceInfo
                        {
                            Id = name,
                            Name = description,
                            Channels = 2, // 假设支持立体声
                            SupportedSampleRates = new[] { 8000, 11025, 16000, 22050, 24000, 44100, 48000, 96000 },
                            IsDefault = name == "default" || name.Contains("default")
                        });
                    }

                    // 移动到下一个设备
                    currentHint = IntPtr.Add(currentHint, IntPtr.Size);
                    if (Marshal.ReadIntPtr(currentHint) == IntPtr.Zero)
                        break;
                }

                AlsaNative.snd_device_name_free_hint(hints);
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"Failed to enumerate ALSA devices: {ex.Message}"));
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
            if (_alsaHandle != IntPtr.Zero)
            {
                CloseDevice();
            }

            string deviceName = _currentDevice?.Id ?? "default";

            try
            {
                int result = AlsaNative.snd_pcm_open(out _alsaHandle, deviceName,
                    AlsaNative.SND_PCM_STREAM_PLAYBACK, 0);

                if (result < 0)
                {
                    OnErrorOccurred(new Exception($"ALSA open failed: {result}"));
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"ALSA initialization failed: {ex.Message}"));
                return false;
            }
        }

        protected override bool PlaySampleInternal(SampleEntity sample)
        {
            if (_alsaHandle == IntPtr.Zero || sample.Sample == null || sample.Sample.Length == 0)
                return false;

            try
            {
                // 检查是否需要重新配置设备
                if (sample.SampleRate != _currentSampleRate || sample.Channels != _currentChannels)
                {
                    if (!ConfigureAlsaDevice(sample.SampleRate, sample.Channels))
                        return false;
                }

                int size = sample.Sample.Length;
                IntPtr buffer = Marshal.AllocHGlobal(size * 4); // 4 bytes per float

                try
                {
                    Marshal.Copy(sample.Sample, 0, buffer, size);
                    int result = AlsaNative.snd_pcm_writei(_alsaHandle, buffer, size);

                    if (result < 0)
                    {
                        // 尝试恢复设备
                        result = AlsaNative.snd_pcm_recover(_alsaHandle, result, 1);
                        if (result >= 0)
                        {
                            result = AlsaNative.snd_pcm_writei(_alsaHandle, buffer, size);
                        }
                    }

                    return result >= 0;
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"ALSA playback failed: {ex.Message}"), sample);
                return false;
            }
        }

        protected override void CloseDevice()
        {
            if (_alsaHandle != IntPtr.Zero)
            {
                try
                {
                    AlsaNative.snd_pcm_drain(_alsaHandle);
                    AlsaNative.snd_pcm_close(_alsaHandle);
                }
                catch (Exception ex)
                {
                    OnErrorOccurred(new Exception($"ALSA close failed: {ex.Message}"));
                }
                finally
                {
                    _alsaHandle = IntPtr.Zero;
                    _currentSampleRate = 0;
                    _currentChannels = 0;
                    _isPaused = false;
                }
            }
        }

        protected override void PauseInternal()
        {
            if (_alsaHandle != IntPtr.Zero && !_isPaused)
            {
                int result = AlsaNative.snd_pcm_pause(_alsaHandle, 1);
                if (result >= 0)
                {
                    _isPaused = true;
                }
            }
        }

        protected override void ResumeInternal()
        {
            if (_alsaHandle != IntPtr.Zero && _isPaused)
            {
                int result = AlsaNative.snd_pcm_pause(_alsaHandle, 0);
                if (result < 0)
                {
                    // 如果暂停恢复失败，尝试准备设备
                    result = AlsaNative.snd_pcm_prepare(_alsaHandle);
                }

                if (result >= 0)
                {
                    _isPaused = false;
                }
            }
        }
        #endregion

        #region 状态检查实现
        public override bool IsDeviceReady()
        {
            return _alsaHandle != IntPtr.Zero;
        }

        public override bool IsFormatSupported(int sampleRate, int channels)
        {
            // ALSA 通常支持常见的采样率和通道数
            int[] supportedRates = { 8000, 11025, 16000, 22050, 24000, 44100, 48000, 96000 };
            return supportedRates.Contains(sampleRate) && (channels == 1 || channels == 2);
        }

        public override int GetDeviceLatency()
        {
            // 返回估计的延迟（毫秒）
            if (_currentSampleRate > 0)
            {
                // 假设缓冲区大小为1024样本
                return (1024 * 1000) / _currentSampleRate;
            }
            return 50; // 默认50ms
        }
        #endregion

        #region 辅助方法
        private bool ConfigureAlsaDevice(int sampleRate, int channels)
        {
            if (_alsaHandle == IntPtr.Zero)
                return false;

            try
            {
                int result = AlsaNative.snd_pcm_set_params(_alsaHandle,
                    AlsaNative.SND_PCM_FORMAT_FLOAT_LE,
                    AlsaNative.SND_PCM_ACCESS_RW_INTERLEAVED,
                    channels,
                    sampleRate,
                    1, // soft_resample
                    100000); // latency in microseconds (100ms)

                if (result >= 0)
                {
                    _currentSampleRate = sampleRate;
                    _currentChannels = channels;
                    return true;
                }

                OnErrorOccurred(new Exception($"ALSA configuration failed: {result}"));
                return false;
            }
            catch (Exception ex)
            {
                OnErrorOccurred(new Exception($"ALSA configuration error: {ex.Message}"));
                return false;
            }
        }

        // ALSA恢复函数（需要声明）
        [DllImport("libasound.so.2")]
        private static extern int snd_pcm_recover(IntPtr pcm, int err, int silent);
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