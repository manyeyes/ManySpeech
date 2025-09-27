using AudioInOut.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AudioInOut.Base
{
    public static class MathHelper
    {
        /// <summary>
        /// 将值限制在指定的最小值和最大值之间
        /// </summary>
        /// <typeparam name="T">可比较的数值类型（如 float、double、int 等）</typeparam>
        /// <param name="value">要限制的值</param>
        /// <param name="min">最小值</param>
        /// <param name="max">最大值</param>
        /// <returns>限制后的值（若 value 小于 min 则返回 min；若大于 max 则返回 max；否则返回 value）</returns>
        public static T Clamp<T>(T value, T min, T max) where T : IComparable<T>
        {
            // 如果值小于最小值，返回最小值
            if (value.CompareTo(min) < 0)
                return min;
            // 如果值大于最大值，返回最大值
            if (value.CompareTo(max) > 0)
                return max;
            // 否则返回原值
            return value;
        }
    }
    /// <summary>
    /// 音频播放器抽象基类
    /// </summary>
    public abstract class BasePlayer : IPlayer
    {
        #region 字段
        protected readonly LinkedList<SampleEntity> _sampleQueue = new LinkedList<SampleEntity>();
        protected Func<SampleEntity?>? _sampleProvider;
        protected Action<string>? _messageHandler;
        protected AudioDeviceInfo? _currentDevice;
        protected CancellationTokenSource? _playbackCts;
        protected Task? _playbackTask;
        protected readonly object _lockObject = new object();
        protected float _volume = 1.0f;
        protected int _bufferSizeMs = 100;
        #endregion

        #region 属性实现
        public abstract AudioPlaybackState State { get; }
        public abstract bool IsActivated { get; }
        public AudioDeviceInfo? CurrentDevice => _currentDevice;
        public abstract string Name { get; }

        public virtual float Volume
        {
            get => _volume;
#if NET6_0_OR_GREATER && NETSTANDARD2_1 && NETCOREAPP3_0_OR_GREATER
            set => _volume = Math.Clamp(value, 0.0f, 1.0f);
#else
            set => _volume = MathHelper.Clamp(value, 0.0f, 1.0f);
#endif
        }

        public virtual int BufferSizeMs
        {
            get => _bufferSizeMs;
            set => _bufferSizeMs = Math.Max(10, value);
        }
        #endregion

        #region 事件实现
        public event AudioPlaybackEventHandler? PlaybackStarted;
        public event AudioPlaybackEventHandler? PlaybackStopped;
        public event AudioPlaybackEventHandler? PlaybackCompleted;
        public event AudioPlaybackEventHandler? ErrorOccurred;

        protected virtual void OnPlaybackStarted(SampleEntity? sample = null)
        {
            PlaybackStarted?.Invoke(this, new AudioPlaybackEventArgs
            {
                Sample = sample,
                State = AudioPlaybackState.Playing
            });
        }

        protected virtual void OnPlaybackStopped(SampleEntity? sample = null)
        {
            PlaybackStopped?.Invoke(this, new AudioPlaybackEventArgs
            {
                Sample = sample,
                State = AudioPlaybackState.Stopped
            });
        }

        protected virtual void OnPlaybackCompleted(SampleEntity? sample = null)
        {
            PlaybackCompleted?.Invoke(this, new AudioPlaybackEventArgs
            {
                Sample = sample,
                State = AudioPlaybackState.Stopped
            });
        }

        protected virtual void OnErrorOccurred(Exception error, SampleEntity? sample = null)
        {
            ErrorOccurred?.Invoke(this, new AudioPlaybackEventArgs
            {
                Sample = sample,
                State = State,
                Error = error
            });
        }
        #endregion

        #region 抽象方法 - 子类必须实现
        protected abstract bool InitializeDevice();
        protected abstract bool PlaySampleInternal(SampleEntity sample);
        protected abstract void CloseDevice();
        protected abstract void PauseInternal();
        protected abstract void ResumeInternal();
        #endregion

        #region 设备管理实现
        public abstract IReadOnlyList<AudioDeviceInfo> GetAudioDevices();

        public virtual bool SelectDevice(string deviceId)
        {
            var devices = GetAudioDevices();
            var device = devices.FirstOrDefault(d => d.Id == deviceId);

            if (device != null)
            {
                _currentDevice = device;
                return true;
            }

            return false;
        }

        public virtual bool SelectDefaultDevice()
        {
            var devices = GetAudioDevices();
            var defaultDevice = devices.FirstOrDefault(d => d.IsDefault) ?? devices.FirstOrDefault();

            if (defaultDevice != null)
            {
                _currentDevice = defaultDevice;
                return true;
            }

            return false;
        }
        #endregion

        #region 播放控制实现
        public virtual async Task PlayAsync(CancellationToken cancellationToken = default)
        {
            if (State == AudioPlaybackState.Playing)
                return;

            lock (_lockObject)
            {
                if (_playbackCts != null && !_playbackCts.IsCancellationRequested)
                {
                    _playbackCts.Cancel();
                    _playbackCts.Dispose();
                }

                _playbackCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            }

            if (!InitializeDevice())
            {
                OnErrorOccurred(new InvalidOperationException("Failed to initialize audio device"));
                return;
            }

            _playbackTask = Task.Run(() => PlaybackLoop(_playbackCts.Token), _playbackCts.Token);
            await Task.CompletedTask;
        }

        public virtual void Pause()
        {
            if (State == AudioPlaybackState.Playing)
            {
                PauseInternal();
            }
        }

        public virtual void Resume()
        {
            if (State == AudioPlaybackState.Paused)
            {
                ResumeInternal();
            }
        }

        public virtual void Stop()
        {
            lock (_lockObject)
            {
                _playbackCts?.Cancel();
                _playbackCts?.Dispose();
                _playbackCts = null;
            }

            CloseDevice();
            ClearQueue();
        }

        public virtual async Task WaitForCompletionAsync(CancellationToken cancellationToken = default)
        {
            if (_playbackTask != null)
            {
                try
                {
                    await _playbackTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // 任务被取消是正常的
                }
            }
        }
        #endregion

        #region 样本管理实现
        public virtual void AddSample(SampleEntity sample)
        {
            lock (_lockObject)
            {
                _sampleQueue.AddLast(sample);
            }
        }

        public virtual void AddSamples(IEnumerable<SampleEntity> samples)
        {
            lock (_lockObject)
            {
                foreach (var sample in samples)
                {
                    _sampleQueue.AddLast(sample);
                }
            }
        }

        public virtual int GetQueueCount()
        {
            lock (_lockObject)
            {
                return _sampleQueue.Count;
            }
        }

        public virtual void ClearQueue()
        {
            lock (_lockObject)
            {
                _sampleQueue.Clear();
            }
        }

        public virtual void SetSampleProvider(Func<SampleEntity?> sampleProvider)
        {
            _sampleProvider = sampleProvider;
        }

        public virtual void SetMessageHandler(Action<string> messageHandler)
        {
            _messageHandler = messageHandler;
        }
        #endregion

        #region 状态检查实现
        public abstract bool IsDeviceReady();
        public abstract bool IsFormatSupported(int sampleRate, int channels);
        public abstract int GetDeviceLatency();
        #endregion

        #region 播放循环
        protected virtual async Task PlaybackLoop(CancellationToken cancellationToken)
        {
            OnPlaybackStarted();

            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    var sample = GetNextSample();
                    if (sample == null)
                    {
                        await Task.Delay(10, cancellationToken);
                        continue;
                    }

                    if (!IsFormatSupported(sample.SampleRate, sample.Channels))
                    {
                        OnErrorOccurred(new InvalidOperationException(
                            $"Unsupported audio format: {sample.SampleRate}Hz, {sample.Channels} channels"), sample);
                        continue;
                    }

                    _messageHandler?.Invoke(sample.Text ?? string.Empty);

                    if (PlaySampleInternal(sample))
                    {
                        //await Task.Delay(CalculateDelay(sample), cancellationToken);
                        await Task.Delay(50, cancellationToken);
                    }
                    else
                    {
                        OnErrorOccurred(new InvalidOperationException("Failed to play sample"), sample);
                    }

                    if (cancellationToken.IsCancellationRequested)
                        break;
                }
            }
            catch (OperationCanceledException)
            {
                // 正常取消
            }
            catch (Exception ex)
            {
                OnErrorOccurred(ex);
            }
            finally
            {
                OnPlaybackCompleted();
                CloseDevice();
            }
        }

        protected virtual SampleEntity? GetNextSample()
        {
            lock (_lockObject)
            {
                if (_sampleProvider != null)
                {
                    return _sampleProvider();
                }

                if (_sampleQueue.Count > 0)
                {
                    var sample = _sampleQueue.First.Value;
                    _sampleQueue.RemoveFirst();
                    return sample;
                }
            }

            return null;
        }

        protected virtual int CalculateDelay(SampleEntity sample)
        {
            if (sample.Sample == null || sample.Sample.Length == 0)
                return 10;

            // 根据样本长度和采样率计算延迟
            var durationMs = (sample.Sample.Length * 1000) / sample.SampleRate;
            return Math.Max(10, durationMs);
        }
        #endregion

        #region IDisposable实现
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Stop();
                ClearQueue();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}