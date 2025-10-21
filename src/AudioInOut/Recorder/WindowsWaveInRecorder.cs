using AudioInOut.Base;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using static AudioInOut.Recorder.WaveInterop;

namespace AudioInOut.Recorder
{
    public class WindowsWaveInRecorder : BaseRecorder, IDisposable
    {
        private bool _disposed = false;
        private readonly object _disposeLock = new object();

        private nint _waveInHandle = default(nint);
        private readonly List<nint> _bufferPointers = new();
        private readonly List<nint> _headerPointers = new();
        private readonly ConcurrentQueue<float[]> _audioChunkQueue;

        public bool IsCapturing { get; private set; }
        public const int BitsPerSample = 16;
        public const int Channels = 1;
        public const int SampleRate = 16000;
        //private readonly int _sampleRate;
        private readonly int _bufferMilliseconds;

        // 定义回调委托
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void WaveInCallbackDelegate(nint hwi, WaveMessage uMsg, nint dwInstance, nint dwParam1, nint dwParam2);

        private readonly WaveInProc _callbackDelegate;
        private readonly GCHandle _callbackHandle;

        public WindowsWaveInRecorder(int bufferMilliseconds = 100)
        {
            //_sampleRate = 16000;
            _bufferMilliseconds = bufferMilliseconds;
            _audioChunkQueue = new ConcurrentQueue<float[]>();

            // 创建并固定回调委托
            _callbackDelegate = WaveInCallbackHandler;
            _callbackHandle = GCHandle.Alloc(_callbackDelegate);

            InitializeWaveDevice();
        }

        private void InitializeWaveDevice()
        {
            var waveFormat = new WaveFormat
            {
                wFormatTag = 1, // WAVE_FORMAT_PCM
                nChannels = Channels,
                nSamplesPerSec = SampleRate,
                wBitsPerSample = BitsPerSample,
                nBlockAlign = Channels * (BitsPerSample / 8),
                nAvgBytesPerSec = SampleRate * Channels * (BitsPerSample / 8),
                cbSize = 0
            };

            Console.WriteLine($"Opening wave device with format: {SampleRate}Hz, {BitsPerSample}bit, {Channels}ch");

            // 使用默认设备 (-1)
            var result = waveInOpen(
                out _waveInHandle,
                -1, // 使用默认设备
                ref waveFormat,
                _callbackDelegate,
                default(nint),
                CALLBACK_FUNCTION);

            Console.WriteLine($"waveInOpen result: {result}");

            if (result != MMSYSERR_NOERROR)
            {
                throw new InvalidOperationException($"Failed to open wave device. Error code: {result}");
            }
        }

        private void WaveInCallbackHandler(nint hwi, WaveMessage uMsg, nint dwInstance, nint dwParam1, nint dwParam2)
        {
            var message = uMsg;
            // Console.WriteLine($"Callback received message: {message}");

            if (message == WaveMessage.WaveInData && IsCapturing)
            {
                ProcessAudioData(dwParam1);
            }
        }

        private void ProcessAudioData(nint headerPtr)
        {
            try
            {
                if (headerPtr == default(nint))
                {
                    return;
                }

                var header = Marshal.PtrToStructure<WaveHeader>(headerPtr);

                if (header.dwBytesRecorded > 0 && header.lpData != default(nint) && IsCapturing)
                {
                    byte[] audioData = new byte[header.dwBytesRecorded];
                    Marshal.Copy(header.lpData, audioData, 0, header.dwBytesRecorded);

                    //audioData = audioData.Select(x => x=0).ToArray();//用于测试，如果没有人声，重新赋值，使得vad不计入人声，但是试验效果不理想，考虑将其放入 SpeechProcessing 中
                    // 转换为浮点数组
                    float[] normalizedSamples = ConvertToFloatSamples(audioData, header.dwBytesRecorded);
                    //float epsilon = 1e-3f * 32768f;
                    //// 填充极小值（可选择交替符号避免直流偏移）
                    //for (int i = 0; i < normalizedSamples.Length; i++)
                    //{
                    //    // 每隔一个采样点取反，减少直流分量
                    //    normalizedSamples[i] = (i % 2 == 0) ? epsilon : -epsilon;
                    //}
                    _audioChunkQueue.Enqueue(normalizedSamples);

                    // 重新提交缓冲区
                    var result = waveInAddBuffer(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>());
                    if (result != MMSYSERR_NOERROR)
                    {
                        Console.WriteLine($"Warning: Failed to re-add buffer. Error: {result}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in audio callback: {ex.Message}");
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
        private void PrepareBuffers()
        {
            // 计算缓冲区大小
            int bufferSize = SampleRate * _bufferMilliseconds * (BitsPerSample / 8) * Channels / 1000;
            const int bufferCount = 3; // 三重缓冲
            CleanupBuffers();
            for (int i = 0; i < bufferCount; i++)
            {
                // 分配音频数据缓冲区
                nint dataPtr = Marshal.AllocHGlobal(bufferSize);
                _bufferPointers.Add(dataPtr);

                // 创建 wave header
                var header = new WaveHeader
                {
                    lpData = dataPtr,
                    dwBufferLength = bufferSize,
                    dwBytesRecorded = 0,
                    dwFlags = 0,
                    dwLoops = 0,
                    lpNext = default(nint),
                    reserved = default(nint)
                };

                // 分配并初始化 header
                nint headerPtr = Marshal.AllocHGlobal(Marshal.SizeOf<WaveHeader>());
                Marshal.StructureToPtr(header, headerPtr, false);
                _headerPointers.Add(headerPtr);
                try
                {
                    // 准备 header
                    var prepareResult = waveInPrepareHeader(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>());

                    if (prepareResult != MMSYSERR_NOERROR)
                    {
                        Console.WriteLine($"Warning: Prepare header failed. Error: {prepareResult}");
                        continue;
                    }

                    // 添加缓冲区
                    var addResult = waveInAddBuffer(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>());

                    if (addResult != MMSYSERR_NOERROR)
                    {
                        //waveInUnprepareHeader(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>()); // 释放已准备的header
                        //CleanupBuffers();
                        //throw new InvalidOperationException($"Failed to add buffer (index {i}): {errorMsg} (Code: {addResult})");
                        Console.WriteLine($"Warning: Add buffer failed. Error: {addResult}");
                    }
                }
                catch (Exception ex)
                {
                    // 发生错误时，清理当前循环已分配的资源
                    Console.WriteLine($"Error in PrepareBuffers (index {i}): {ex.Message}");

                    // 移除当前循环添加的指针（避免 Cleanup 释放未分配的资源）
                    if (dataPtr != default(nint) && _bufferPointers.Contains(dataPtr))
                    {
                        _bufferPointers.Remove(dataPtr);
                        Marshal.FreeHGlobal(dataPtr);
                    }
                    if (headerPtr != default(nint) && _headerPointers.Contains(headerPtr))
                    {
                        _headerPointers.Remove(headerPtr);
                        waveInUnprepareHeader(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>());
                        Marshal.FreeHGlobal(headerPtr);
                    }

                    // 彻底清理所有已分配的资源
                    CleanupBuffers();

                    // 终止循环，不再继续分配
                    throw; // 抛出异常，让上层处理（如终止启动）
                }
            }
        }

        public override async Task StartCapture()
        {
            if (IsCapturing) return;

            lock (_disposeLock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(WindowsWaveInRecorder));

                // 准备音频缓冲区
                PrepareBuffers();

                // 开始录音
                var result = waveInStart(_waveInHandle);
                if (result != MMSYSERR_NOERROR)
                {
                    throw new InvalidOperationException($"Failed to start recording. Error: {result}");
                }
                IsCapturing = true;
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] 麦克风实时采集已启动，按 ESC 键停止...");
            }
        }

        public override void StopCapture()
        {
            if (!IsCapturing) return;

            lock (_disposeLock)
            {
                IsCapturing = false;

                // 停止录音
                var stopResult = waveInStop(_waveInHandle);
                if (stopResult != MMSYSERR_NOERROR)
                {
                    Console.WriteLine($"Warning: Stop failed. Error: {stopResult}");
                }
                waveInReset(_waveInHandle); // 清空驱动队列
                // 清理缓冲区
                CleanupBuffers();

                // 发送结束信号
                _audioChunkQueue.Enqueue(null);

                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] 麦克风实时采集已停止");
            }
        }

        private void CleanupBuffers()
        {
            foreach (var headerPtr in _headerPointers)
            {
                if (headerPtr != default(nint))
                {
                    var unprepareResult = waveInUnprepareHeader(_waveInHandle, headerPtr, Marshal.SizeOf<WaveHeader>());

                    if (unprepareResult != MMSYSERR_NOERROR)
                    {
                        Console.WriteLine($"Warning: Unprepare header failed. Error: {unprepareResult}");
                    }

                    Marshal.FreeHGlobal(headerPtr);
                }
            }

            foreach (var dataPtr in _bufferPointers)
            {
                if (dataPtr != default(nint))
                {
                    Marshal.FreeHGlobal(dataPtr);
                }
            }

            _headerPointers.Clear();
            _bufferPointers.Clear();
        }

        public override async Task<List<List<float[]>>?> GetNextMicChunkAsync(CancellationToken cancellationToken)
        {
            try
            {
                while (IsCapturing && _audioChunkQueue.IsEmpty)
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
            catch (TaskCanceledException ex)
            {
                // 区分：是当前令牌取消，还是其他令牌取消（如Task.Delay的令牌）
                if (cancellationToken.IsCancellationRequested)
                {
                    // 正常取消：返回null，符合原有逻辑
                    //Console.WriteLine("GetNextMicChunkAsync: 任务被正常取消，返回null");
                    Console.WriteLine("The task of get mic data: cancelled normally，return null");
                    return null;
                }
                else
                {
                    // 意外取消（如其他令牌）：重新抛出异常
                    //Console.WriteLine($"GetNextMicChunkAsync: 意外取消 - {ex.Message}");
                    Console.WriteLine($"The task of get mic data: cancelled unexpectedly - {ex.Message}");
                    throw;
                }
            }
            catch (Exception ex)
            {
                // 其他异常（如队列操作异常）：按需处理
                //Console.WriteLine($"GetNextMicChunkAsync: 其他异常 - {ex.Message}");
                Console.WriteLine($"The task of get mic data: other exceptions - {ex.Message}");
                return null;
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            lock (_disposeLock)
            {
                if (_disposed) return;

                if (disposing)
                {
                    // 停止采集
                    StopCapture();
                }

                // 释放非托管资源
                if (_waveInHandle != default(nint))
                {
                    var closeResult = waveInClose(_waveInHandle);
                    if (closeResult != MMSYSERR_NOERROR)
                    {
                        Console.WriteLine($"Warning: Close failed. Error: {closeResult}");
                    }
                    _waveInHandle = default(nint);
                }

                if (_callbackHandle.IsAllocated)
                {
                    _callbackHandle.Free();
                }

                _disposed = true;
            }
        }

        public override void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~WindowsWaveInRecorder()
        {
            Dispose(false);
        }
    }

    // Windows 互操作封装
    internal static class WaveInterop
    {
        public const int MMSYSERR_NOERROR = 0;
        public const int CALLBACK_FUNCTION = 0x30000;

        public enum WaveMessage : uint
        {
            WaveInOpen = 0x3BE,
            WaveInClose = 0x3BF,
            WaveInData = 0x3C0
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        public struct WaveInCapabilities
        {
            public ushort wMid;
            public ushort wPid;
            public uint vDriverVersion;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
            public string szPname;
            public uint dwFormats;
            public ushort wChannels;
            public ushort wReserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WaveFormat
        {
            public short wFormatTag;
            public short nChannels;
            public int nSamplesPerSec;
            public int nAvgBytesPerSec;
            public short nBlockAlign;
            public short wBitsPerSample;
            public short cbSize;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct WaveHeader
        {
            public nint lpData;
            public int dwBufferLength;
            public int dwBytesRecorded;
            public nint dwUser;
            public int dwFlags;
            public int dwLoops;
            public nint lpNext;
            public nint reserved;
        }

        [DllImport("winmm.dll", EntryPoint = "waveInGetNumDevs")]
        public static extern int waveInGetNumDevs();

        [DllImport("winmm.dll", EntryPoint = "waveInGetDevCapsW", CharSet = CharSet.Unicode)]
        public static extern int waveInGetDevCaps(int uDeviceID, ref WaveInCapabilities lpCaps, int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveInOpen")]
        public static extern int waveInOpen(out nint phwi, int uDeviceID,
            ref WaveFormat pwfx, WaveInProc dwCallback, nint dwInstance, int fdwOpen);

        [DllImport("winmm.dll", EntryPoint = "waveInClose")]
        public static extern int waveInClose(nint hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInStart")]
        public static extern int waveInStart(nint hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInStop")]
        public static extern int waveInStop(nint hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInReset")]
        public static extern int waveInReset(nint hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInPrepareHeader")]
        public static extern int waveInPrepareHeader(nint hwi, nint pwh, int cbwh);

        [DllImport("winmm.dll", EntryPoint = "waveInUnprepareHeader")]
        public static extern int waveInUnprepareHeader(nint hwi, nint pwh, int cbwh);

        [DllImport("winmm.dll", EntryPoint = "waveInAddBuffer")]
        public static extern int waveInAddBuffer(nint hwi, nint pwh, int cbwh);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void WaveInProc(nint hwi, WaveMessage uMsg, nint dwInstance,
            nint dwParam1, nint dwParam2);
    }
}
