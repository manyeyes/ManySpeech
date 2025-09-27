// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
using ManySpeech.AudioSep.Model;
using SpeechFeatures;

namespace ManySpeech.AudioSep
{
    /// <summary>
    /// WavFrontend
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        private OnlineFbank _onlineFbank;
        private const double EPS = 1e-6; // 定义 EPS 常量，用于数值稳定性

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                window_type: _frontendConfEntity.window,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                frame_shift: _frontendConfEntity.frame_shift,
                frame_length: _frontendConfEntity.frame_length
                );
        }

        public float[] GetFbank(float[] samples)
        {
            float sample_rate = _frontendConfEntity.fs;
            float[] fbanks = _onlineFbank.GetFbank(samples);
            return fbanks;
        }

        public static (double[] normalizedAudio, double scaleFactor) AudioNorm(double[] x)
        {
            // 计算输入音频信号的均方根(RMS)
            double sumSquares = 0;
            foreach (var value in x)
            {
                sumSquares += value * value;
            }
            double rms = Math.Sqrt(sumSquares / x.Length);

            // 计算将信号调整到目标电平(-25 dB)的标量
            double scalar = Math.Pow(10, -25 / 20.0) / (rms + EPS);

            // 应用第一阶段缩放
            double[] scaledX = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                scaledX[i] = x[i] * scalar;
            }

            // 计算缩放后音频信号的功率
            double[] powX = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                powX[i] = scaledX[i] * scaledX[i];
            }

            // 计算平均功率
            double avgPowX = 0;
            foreach (var value in powX)
            {
                avgPowX += value;
            }
            avgPowX /= x.Length;

            // 仅针对高于平均功率的音频片段计算RMS
            double sumHighPowerSquares = 0;
            int highPowerCount = 0;
            foreach (var value in powX)
            {
                if (value > avgPowX)
                {
                    sumHighPowerSquares += value;
                    highPowerCount++;
                }
            }

            double rmsx = 0;
            if (highPowerCount > 0)
            {
                rmsx = Math.Sqrt(sumHighPowerSquares / highPowerCount);
            }

            // 计算第二个标量，用于基于高功率片段进一步归一化
            double scalarx = Math.Pow(10, -25 / 20.0) / (rmsx + EPS);

            // 应用第二阶段缩放
            double[] finalX = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                finalX[i] = scaledX[i] * scalarx;
            }

            // 返回双归一化后的音频信号和总缩放因子的倒数
            return (finalX, 1 / (scalar * scalarx + EPS));
        }
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_onlineFbank != null)
                {
                    _onlineFbank.Dispose();
                }
                if (_frontendConfEntity != null)
                {
                    _frontendConfEntity = null;
                }
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
