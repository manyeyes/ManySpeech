using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;
using System.Numerics;

namespace ManySpeech.AudioSep.Utils
{

    public class AudioBandwidthProcessor
    {
        public static float[] BandwidthSub(float[] lowBandwidthAudio, float[] highBandwidthAudio, int fs = 48000)
        {
            // Detect effective bandwidth of the first signal
            var (fLow, fHigh) = DetectBandwidth(lowBandwidthAudio, fs);

            // Replace the lower frequency of the second audio
            float[] substitutedAudio = ReplaceBandwidth(lowBandwidthAudio, highBandwidthAudio, fs, fLow, fHigh);

            // Optional: Smooth the transition
            float[] smoothedAudio = substitutedAudio;// SmoothTransition(substitutedAudio, lowBandwidthAudio, fs);
            return smoothedAudio;
        }

        private static (double fLow, double fHigh) DetectBandwidth(float[] signal, int fs, float energyThreshold = 0.99f)
        {
            // Perform STFT
            var stftResult = SignalProcessing.Stft(signal.Select(x=>(double)x).ToArray(), fs);
            var f = stftResult.frequencies;
            var Zxx = stftResult.spectrogram;

            // Calculate power spectral density
            //var psd = Zxx.PointwisePower(2);
            // 计算复数矩阵 Zxx 的 PSD (|Zxx|²)
            var psd = ComputePSDOptimized(Zxx);
            // 或者分步实现：
            //var magnitudeSquared = Zxx.PointwiseMultiply(Zxx.Conjugate());
            //var psd = magnitudeSquared.Real(); // 因为 |z|² = z * z̅ 总是实数
            // Calculate total energy and cumulative energy
            double totalEnergy = psd.Enumerate().Sum(x => x);
            var cumulativeEnergy = new double[f.Length];
            double sum = 0;

            for (int i = 0; i < f.Length; i++)
            {
                //sum += psd.Row(i).Sum(x => x.Real);
                sum += psd.Row(i).Sum(x => x);
                cumulativeEnergy[i] = sum / totalEnergy;
            }

            // Exclude DC component (0 Hz)
            var validIndices = Enumerable.Range(0, f.Length)
                .Where(i => f[i] > 0)
                .ToArray();

            // Find low frequency bound with safe default
            double fLow = 0;
            try
            {
                var firstAboveThreshold = validIndices.FirstOrDefault(i => cumulativeEnergy[i] > (1 - energyThreshold));
                fLow = firstAboveThreshold != 0 ? f[firstAboveThreshold] : f[validIndices[0]];
            }
            catch
            {
                fLow = f[validIndices[0]];
            }

            // Find high frequency bound with safe default
            double fHigh = fs / 2;
            try
            {
                var firstAtThreshold = validIndices.FirstOrDefault(i => cumulativeEnergy[i] >= energyThreshold);
                fHigh = firstAtThreshold != 0 ? f[firstAtThreshold] : f[validIndices.Last()];
            }
            catch
            {
                fHigh = f[validIndices.Last()];
            }

            // Ensure reasonable bounds
            fLow = Math.Max(fLow, 20); // At least 20Hz
            fHigh = Math.Min(fHigh, fs / 2); // No more than Nyquist

            return (fLow, fHigh);
        }

        public static Matrix<double> ComputePSDOptimized(Matrix<Complex> stftMatrix)
        {
            var psd = Matrix<double>.Build.Dense(stftMatrix.RowCount, stftMatrix.ColumnCount);

            for (int i = 0; i < stftMatrix.RowCount; i++)
            {
                for (int j = 0; j < stftMatrix.ColumnCount; j++)
                {
                    var c = stftMatrix[i, j];
                    psd[i, j] = c.Real * c.Real + c.Imaginary * c.Imaginary;
                }
            }

            return psd;
        }

        private static float[] ReplaceBandwidth(float[] signal1, float[] signal2, int fs, double fLow, double fHigh)
        {
            // Extract effective band from signal1 (lowpass)
            float[] effectiveBand = LowpassFilter(signal1, fs, fHigh);

            // Extract highpass band from signal2
            float[] signal2Highpass = HighpassFilter(signal2, fs, fHigh);

            // Match lengths of the two signals
            int minLength = Math.Min(effectiveBand.Length, signal2Highpass.Length);
            effectiveBand = effectiveBand.Take(minLength).ToArray();
            signal2Highpass = signal2Highpass.Take(minLength).ToArray();

            // Combine the two signals
            float[] result = new float[minLength];
            for (int i = 0; i < minLength; i++)
            {
                result[i] = signal2Highpass[i] + effectiveBand[i];
            }

            return result;
        }

        private static float[] SmoothTransition(float[] signal1, float[] signal2, int fs, float transitionBand = 100)
        {
            int fadeLength = (int)(transitionBand * fs / 1000);
            double[] fade = Generate.LinearSpaced(fadeLength, 0f, 1f);

            int minLength = Math.Min(signal1.Length, signal2.Length);
            float[] crossfade = new float[minLength];

            // Create crossfade array
            for (int i = 0; i < minLength; i++)
            {
                crossfade[i] = i < fadeLength ? (float)fade[i] : 1f;
            }

            // Apply crossfade
            float[] smoothedSignal = new float[minLength];
            for (int i = 0; i < minLength; i++)
            {
                smoothedSignal[i] = (1 - crossfade[i]) * signal2[i] + crossfade[i] * signal1[i];
            }

            return smoothedSignal;
        }

        private static float[] LowpassFilter(float[] signal, int fs, double cutoff)
        {
            double nyquist = 0.5f * fs;
            double cutoffNormalized = cutoff / nyquist;
            var (b, a) = Butterworth(4, cutoffNormalized, "low");
            //b = new double[] { 0.00141968, 0.00567874, 0.0085181, 0.00567874, 0.00141968 };
            //a = new double[] { 1.0,         - 2.84978905,  3.16617803, - 1.60539465,  0.31172061 };
            return Filter(b, a, signal);
        }

        private static float[] HighpassFilter(float[] signal, int fs, double cutoff)
        {
            double nyquist = 0.5 * fs;
            double cutoffNormalized = cutoff / nyquist;
            var (b, a) = Butterworth(4, cutoffNormalized, "high");
            //b = new double[] { 0.55831765, -2.23327058, 3.34990587, -2.23327058, 0.55831765 };
            //a = new double[] { 1.0,         - 2.84978905,  3.16617803, - 1.60539465,  0.31172061 };
            return Filter(b, a, signal);
        }

        private static (double[] b, double[] a) Butterworth(int order, double cutoff, string type)
        {
            // 归一化截止频率
            //double nyquist = 0.5 * fs;
            //double normalizedCutoff = cutoff / nyquist;

            double[] Wn = new double[] { cutoff };
            string btype = type == "low" ? "lowpass" : "highpass";
            var filter = SignalProcessing.Butter(
                order,
                Wn,
                btype, analog: false,
    fs: double.NaN);

            //return (filter.Numerator, filter.Denominator);
            return (filter.Item1, filter.Item2);
        }

        private static float[] Filter(double[] b, double[] a, float[] signal)
        {
            double[] doubleSignal = signal.Select(x => (double)x).ToArray();
            double[] filtered = SignalProcessing.FiltFilt(b, a, doubleSignal);
            return filtered.Select(x => (float)x).ToArray();
        }
    }
}
