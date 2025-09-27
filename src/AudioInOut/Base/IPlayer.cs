using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AudioInOut.Base
{
    // 音频播放状态
    public enum AudioPlaybackState
    {
        Stopped,
        Playing,
        Paused
    }

    // 音频设备信息
    public class AudioDeviceInfo
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public int Channels { get; set; }
        public int[] SupportedSampleRates { get; set; } = Array.Empty<int>();
        public bool IsDefault { get; set; }
    }

    public class SampleEntity
    {
        private string? _text;
        private float[]? _sample;
        private int _sampleRate = 22050;
        private int _channels = 1;

        public float[]? Sample { get => _sample; set => _sample = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Channels { get => _channels; set => _channels = value; }
        public string? Text { get => _text; set => _text = value; }
    }

    public delegate SampleEntity GetSample();
    public delegate void ShowMsg(string msg);

    // 音频播放事件参数
    public class AudioPlaybackEventArgs : EventArgs
    {
        public SampleEntity? Sample { get; set; }
        public AudioPlaybackState State { get; set; }
        public Exception? Error { get; set; }
    }

    // 音频播放回调
    public delegate void AudioPlaybackEventHandler(object sender, AudioPlaybackEventArgs e);

    // 统一的音频播放接口
    public interface IPlayer : IDisposable
    {
        #region 属性
        /// <summary>
        /// 播放状态
        /// </summary>
        AudioPlaybackState State { get; }

        /// <summary>
        /// 是否激活
        /// </summary>
        bool IsActivated { get; }

        /// <summary>
        /// 当前设备信息
        /// </summary>
        AudioDeviceInfo? CurrentDevice { get; }

        /// <summary>
        /// 缓冲区大小（毫秒）
        /// </summary>
        int BufferSizeMs { get; set; }

        /// <summary>
        /// 音量 (0.0 - 1.0)
        /// </summary>
        float Volume { get; set; }
        #endregion

        #region 事件
        /// <summary>
        /// 播放开始事件
        /// </summary>
        event AudioPlaybackEventHandler PlaybackStarted;

        /// <summary>
        /// 播放停止事件
        /// </summary>
        event AudioPlaybackEventHandler PlaybackStopped;

        /// <summary>
        /// 播放完成事件
        /// </summary>
        event AudioPlaybackEventHandler PlaybackCompleted;

        /// <summary>
        /// 错误事件
        /// </summary>
        event AudioPlaybackEventHandler ErrorOccurred;
        #endregion

        #region 设备管理
        /// <summary>
        /// 获取所有可用的音频设备
        /// </summary>
        IReadOnlyList<AudioDeviceInfo> GetAudioDevices();

        /// <summary>
        /// 选择音频设备
        /// </summary>
        bool SelectDevice(string deviceId);

        /// <summary>
        /// 选择默认设备
        /// </summary>
        bool SelectDefaultDevice();
        #endregion

        #region 播放控制
        /// <summary>
        /// 开始播放（异步）
        /// </summary>
        Task PlayAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// 暂停播放
        /// </summary>
        void Pause();

        /// <summary>
        /// 恢复播放
        /// </summary>
        void Resume();

        /// <summary>
        /// 停止播放
        /// </summary>
        void Stop();

        /// <summary>
        /// 等待播放完成
        /// </summary>
        Task WaitForCompletionAsync(CancellationToken cancellationToken = default);
        #endregion

        #region 样本管理
        /// <summary>
        /// 添加样本到播放队列
        /// </summary>
        void AddSample(SampleEntity sample);

        /// <summary>
        /// 批量添加样本
        /// </summary>
        void AddSamples(IEnumerable<SampleEntity> samples);

        /// <summary>
        /// 获取队列中的样本数量
        /// </summary>
        int GetQueueCount();

        /// <summary>
        /// 清空播放队列
        /// </summary>
        void ClearQueue();

        /// <summary>
        /// 设置样本提供者（用于实时生成样本）
        /// </summary>
        void SetSampleProvider(Func<SampleEntity?> sampleProvider);

        /// <summary>
        /// 设置消息显示回调
        /// </summary>
        void SetMessageHandler(Action<string> messageHandler);
        #endregion

        #region 状态检查
        /// <summary>
        /// 检查设备是否就绪
        /// </summary>
        bool IsDeviceReady();

        /// <summary>
        /// 检查格式是否支持
        /// </summary>
        bool IsFormatSupported(int sampleRate, int channels);

        /// <summary>
        /// 获取设备延迟（毫秒）
        /// </summary>
        int GetDeviceLatency();
        #endregion
    }
}

