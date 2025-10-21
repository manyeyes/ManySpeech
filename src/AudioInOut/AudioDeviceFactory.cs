using AudioInOut.Base;
using AudioInOut.Recorder;
using AudioInOut.Player;
using System.Runtime.InteropServices;

namespace AudioInOut
{
    /// <summary>
    /// 平台类型
    /// </summary>
    public enum AudioPlatform
    {
        Windows,
        Linux,
        macOS,
        Unknown
    }

    /// <summary>
    /// 音频播放器类型
    /// </summary>
    public enum AudioPlayerType
    {
        Native,    // 使用平台原生API
        Managed    // 使用托管代码实现（如有）
    }

    public static class AudioPlatformDetector
    {
#if NET461_OR_GREATER
        public static AudioPlatform CurrentPlatform
        {
            get
            {
                if (IsWindows())
                    return AudioPlatform.Windows;
                if (IsLinux())
                    return AudioPlatform.Linux;
                if (IsMacOS())
                    return AudioPlatform.macOS;
                return AudioPlatform.Unknown;
            }
        }
#else
        public static AudioPlatform CurrentPlatform
        {
            get
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    return AudioPlatform.Windows;
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                    return AudioPlatform.Linux;
                if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                    return AudioPlatform.macOS;
                return AudioPlatform.Unknown;
            }
        }
#endif
        // 辅助方法：判断是否为 Windows 系统
        private static bool IsWindows()
        {
            // 在 .NET Framework 中，PlatformID.Win32NT 表示 Windows 系统
            return Environment.OSVersion.Platform == PlatformID.Win32NT;
        }
        private static bool IsLinux()
        {
            // .NET Framework 中没有直接的 Linux 枚举，通过字符串判断
            return Environment.OSVersion.Platform == PlatformID.Unix &&
                   !IsMacOS();
        }

        private static bool IsMacOS()
        {
            // macOS 在 .NET Framework 中被识别为 Unix，可通过特殊路径判断
            return Environment.OSVersion.Platform == PlatformID.Unix &&
                   Directory.Exists("/Applications") &&
                   Directory.Exists("/System") &&
                   Directory.Exists("/Users") &&
                   Directory.Exists("/Volumes");
        }
    }
    public static class AudioDeviceFactory
    {
        #region 创建特定采样率的音频采集设备
        /// <summary>
        /// 创建特定采样率的音频采集设备
        /// </summary>
        public static IRecorder CreateAudioCapture(int sampleRate, int bufferMilliseconds = 100) 
        {
            // 这里可以根据需要创建支持不同采样率的设备
            // 目前所有设备都固定为16kHz，如果需要可变采样率，可以修改实现
            return CreateAudioCapture(bufferMilliseconds);
        }
        public static IRecorder CreateAudioCapture(int bufferMilliseconds = 100)
        {
            return AudioPlatformDetector.CurrentPlatform switch
            {
                AudioPlatform.Windows => new WindowsWaveInRecorder(bufferMilliseconds),
                AudioPlatform.Linux => new LinuxAlsaRecorder(bufferMilliseconds),
                AudioPlatform.macOS => new MacCoreAudioRecorder(bufferMilliseconds),
                _ => throw new PlatformNotSupportedException(
                    $"Audio capture not supported on platform: {AudioPlatformDetector.CurrentPlatform}")
            };
        }

        /// <summary>
        /// 获取推荐的缓冲区大小（毫秒）
        /// </summary>
        public static int GetRecommendedBufferSize()
        {
            return AudioPlatformDetector.CurrentPlatform switch
            {
                AudioPlatform.Windows => 100,    // Windows推荐100ms
                AudioPlatform.Linux => 50,       // Linux推荐50ms
                AudioPlatform.macOS => 100,      // macOS推荐100ms
                _ => 100
            };
        }
        #endregion

        #region 创建音频播放设备
        /// <summary>
        /// 创建音频播放设备
        /// </summary>
        public static IPlayer CreateAudioPlayer(AudioPlayerType playerType)
        {
            return playerType switch
            {
                AudioPlayerType.Native => CreateAudioPlayer(),
                //AudioPlayerType.Managed => new ManagedAudioPlayer(),
                _ => throw new ArgumentException($"Unsupported player type: {playerType}")
            };
        }
        public static IPlayer CreateAudioPlayer()
        {
            return AudioPlatformDetector.CurrentPlatform switch
            {
                AudioPlatform.Windows => new WindowsWaveOutPlayer(),
                AudioPlatform.Linux => new LinuxAlsaPlayer(),
                AudioPlatform.macOS => new MacCoreAudioPlayer(),
                _ => throw new PlatformNotSupportedException(
                    $"Audio player not supported on platform: {AudioPlatformDetector.CurrentPlatform}")
            };
        }
        #endregion

        /// <summary>
        /// 获取平台信息
        /// </summary>
        public static string GetPlatformInfo()
        {
            return AudioPlatformDetector.CurrentPlatform switch
            {
                AudioPlatform.Windows => "Windows (WASAPI)",
                AudioPlatform.Linux => "Linux (ALSA)",
                AudioPlatform.macOS => "macOS (CoreAudio)",
                _ => "Unknown Platform"
            };
        }

        /// <summary>
        /// 检查当前平台是否受支持
        /// </summary>
        public static bool IsAudioSupported()
        {
            return AudioPlatformDetector.CurrentPlatform switch
            {
                AudioPlatform.Windows => true,
                AudioPlatform.Linux => true,
                AudioPlatform.macOS => true, 
                _ => false
            };
        }

        
    }
}
