// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
namespace ManySpeech.DolphinAsr
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2026 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private int _sampleRate = 16000;
        private int _speechLength = 30;

        public WavFrontend(int sampleRate, int speechLength)
        {
            _sampleRate = sampleRate;
            _speechLength = speechLength;
        }

        public float[] GetFeatures(float[] samples)
        {
            float[] features = ResizeAudioDuration(samples, _sampleRate, _speechLength);
            return features;
        }
        /// <summary>
        /// Resamples and pads/truncates raw audio data to a fixed length based on target speech duration.
        /// (Resamples and pads/truncates raw audio data to a fixed length according to the target speech duration)
        /// </summary>
        /// <param name="raw">Raw audio PCM data in float format (16-bit PCM normalized to [-1.0, 1.0])</param>
        /// <param name="sampleRate">Audio sample rate in Hz (e.g., 16000, 44100)</param>
        /// <param name="speechLength">Target speech duration in seconds (0 = return original data)</param>
        /// <returns>Normalized audio data with fixed length (padded with 0s or truncated)</returns>
        public float[] ResizeAudioDuration(float[] raw, float sampleRate, float speechLength)
        {
            // Return original data if target duration is 0 (no resizing needed)
            if (speechLength == 0) return raw;

            // Calculate target number of samples based on sample rate and duration
            int targetSampleCount = (int)(sampleRate * speechLength);
            float[] processedAudio;

            if (raw.Length >= targetSampleCount)
            {
                // Truncate to target length - take first N samples
                processedAudio = new float[targetSampleCount];
                Array.Copy(raw, 0, processedAudio, 0, targetSampleCount);
            }
            else
            {
                // Pad with zeros to reach target length - copy original data to start, rest filled with 0.0f
                processedAudio = new float[targetSampleCount];
                Array.Copy(raw, 0, processedAudio, 0, raw.Length);
                // Remaining elements in new array are already initialized to 0.0f by default
            }

            return processedAudio;
        }
        //public float[] ProcessAudio(float[] raw, float sampleRate, float speechLength)
        //{
        //    // 计算目标长度
        //    if (speechLength == 0) return raw;
        //    int target = (int)(sampleRate * speechLength);
        //    float[] speech;

        //    if (raw.Length >= target)
        //    {
        //        // 截取前 target 个元素
        //        speech = new float[target];
        //        Array.Copy(raw, 0, speech, 0, target);
        //    }
        //    else
        //    {
        //        // 创建长度为 target 的数组，并将 raw 复制到开头，剩余部分自动填充为 0
        //        speech = new float[target];
        //        Array.Copy(raw, 0, speech, 0, raw.Length);
        //        // 新数组剩余元素默认已经是 0.0f，无需额外操作
        //    }

        //    return speech;
        //}

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                //
            }
        }
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
