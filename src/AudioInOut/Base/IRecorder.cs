using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AudioInOut.Base
{
    /// <summary>
    /// 音频采集设备统一接口
    /// </summary>
    public interface IRecorder : IDisposable
    {
        /// <summary>
        /// 是否正在采集
        /// </summary>
        bool IsCapturing { get; }

        /// <summary>
        /// 采样率
        /// </summary>
        int SampleRate { get; }

        /// <summary>
        /// 位深度
        /// </summary>
        int BitsPerSample { get; }

        /// <summary>
        /// 通道数
        /// </summary>
        int Channels { get; }

        /// <summary>
        /// 开始音频采集
        /// </summary>
        Task StartCapture();

        /// <summary>
        /// 停止音频采集
        /// </summary>
        void StopCapture();

        /// <summary>
        /// 获取下一个音频数据块
        /// </summary>
        /// <param name="cancellationToken">取消令牌</param>
        /// <returns>音频数据块列表</returns>
        Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken);
    }

    /// <summary>
    /// 音频数据可用事件参数
    /// </summary>
    public class AudioDataAvailableEventArgs : EventArgs
    {
        public byte[] AudioData { get; }
        public int BytesRecorded { get; }

        public AudioDataAvailableEventArgs(byte[] audioData, int bytesRecorded)
        {
            AudioData = audioData;
            BytesRecorded = bytesRecorded;
        }
    }
}

