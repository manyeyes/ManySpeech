using MathNet.Numerics;

namespace ManySpeech.AudioSep.Utils
{
    public class MelFilterBank
    {
        public static float[,] Mel(int sr, int n_fft, int n_mels = 128, float fmin = 0.0f, float? fmax = null,
                                  bool htk = false, string norm = "slaney", string dtype = "float32")
        {
            if (fmax == null)
            {
                fmax = (float)sr / 2;
            }

            // Initialize the weights
            n_mels = (int)n_mels;
            float[,] weights = new float[n_mels, (int)(1 + n_fft / 2)];

            // Center freqs of each FFT bin
            float[] fftfreqs = FftFrequencies(sr, n_fft);

            // 'Center freqs' of mel bands - uniformly spaced between limits
            float[] mel_f = MelFrequencies(n_mels + 2, fmin, fmax.Value, htk);

            float[] fdiff = new float[mel_f.Length - 1];
            for (int i = 0; i < fdiff.Length; i++)
            {
                fdiff[i] = mel_f[i + 1] - mel_f[i];
            }

            // Compute ramps matrix
            float[,] ramps = new float[mel_f.Length, fftfreqs.Length];
            for (int i = 0; i < mel_f.Length; i++)
            {
                for (int j = 0; j < fftfreqs.Length; j++)
                {
                    ramps[i, j] = mel_f[i] - fftfreqs[j];
                }
            }

            for (int i = 0; i < n_mels; i++)
            {
                // lower and upper slopes for all bins
                float[] lower = new float[fftfreqs.Length];
                float[] upper = new float[fftfreqs.Length];

                for (int j = 0; j < fftfreqs.Length; j++)
                {
                    lower[j] = -ramps[i, j] / fdiff[i];
                    upper[j] = ramps[i + 2, j] / fdiff[i + 1];
                }

                // intersect them with each other and zero
                for (int j = 0; j < fftfreqs.Length; j++)
                {
                    weights[i, j] = Math.Max(0, Math.Min(lower[j], upper[j]));
                }
            }

            if (norm == "slaney")
            {
                // Slaney-style mel is scaled to be approx constant energy per channel
                float[] enorm = new float[n_mels];
                for (int i = 0; i < n_mels; i++)
                {
                    enorm[i] = 2.0f / (mel_f[i + 2] - mel_f[i]);
                }

                for (int i = 0; i < n_mels; i++)
                {
                    for (int j = 0; j < fftfreqs.Length; j++)
                    {
                        weights[i, j] *= enorm[i];
                    }
                }
            }
            else
            {
                weights = Normalize(weights, norm, -1);
            }

            // Check for empty channels
            bool hasEmptyChannels = false;
            for (int i = 0; i < n_mels; i++)
            {
                if (mel_f[i] != 0 && weights.GetRow(i).Max() <= 0)
                {
                    hasEmptyChannels = true;
                    break;
                }
            }

            if (hasEmptyChannels)
            {
                Console.WriteLine("Warning: Empty filters detected in mel frequency basis. " +
                                "Some channels will produce empty responses. " +
                                "Try increasing your sampling rate (and fmax) or " +
                                "reducing n_mels.");
            }

            return weights;
        }

        public static float[] FftFrequencies(float sr = 22050, int n_fft = 2048)
        {
            float[] freqs = new float[1 + n_fft / 2];
            for (int i = 0; i < freqs.Length; i++)
            {
                freqs[i] = i * sr / n_fft;
            }
            return freqs;
        }

        public static float[] MelFrequencies(int n_mels = 128, float fmin = 0.0f, float fmax = 11025.0f, bool htk = false)
        {
            // 'Center freqs' of mel bands - uniformly spaced between limits
            float min_mel = HzToMel(fmin, htk);
            float max_mel = HzToMel(fmax, htk);

            double[] mels = Generate.LinearSpaced(n_mels, min_mel, max_mel);

            float[] hz = new float[n_mels];
            for (int i = 0; i < n_mels; i++)
            {
                hz[i] = MelToHz((float)mels[i], htk);
            }
            return hz;
        }

        public static float HzToMel(float frequencies, bool htk = false)
        {
            if (htk)
            {
                return 2595.0f * (float)Math.Log10(1.0 + frequencies / 700.0);
            }

            // Slaney's Auditory Toolbox
            float f_min = 0.0f;
            float f_sp = 200.0f / 3;

            float mel = (frequencies - f_min) / f_sp;

            float min_log_hz = 1000.0f;
            float min_log_mel = (min_log_hz - f_min) / f_sp;
            float logstep = (float)(Math.Log(6.4) / 27.0);

            if (frequencies >= min_log_hz)
            {
                mel = min_log_mel + (float)Math.Log(frequencies / min_log_hz) / logstep;
            }

            return mel;
        }

        public static float MelToHz(float mel, bool htk = false)
        {
            if (htk)
            {
                return 700.0f * ((float)Math.Pow(10.0, mel / 2595.0) - 1.0f);
            }

            // Slaney's Auditory Toolbox
            float f_min = 0.0f;
            float f_sp = 200.0f / 3;
            float freqs = f_min + f_sp * mel;

            float min_log_hz = 1000.0f;
            float min_log_mel = (min_log_hz - f_min) / f_sp;
            float logstep = (float)(Math.Log(6.4) / 27.0);

            if (mel >= min_log_mel)
            {
                freqs = min_log_hz * (float)Math.Exp(logstep * (mel - min_log_mel));
            }

            return freqs;
        }

        private static float[,] Normalize(float[,] matrix, string norm, int axis)
        {
            // Simplified normalization - for full implementation you'd need to handle different norm types
            if (norm == "l1" || norm == "l2" || norm == "max")
            {
                int rows = matrix.GetLength(0);
                int cols = matrix.GetLength(1);
                float[,] result = new float[rows, cols];

                if (axis == -1 || axis == 1)
                {
                    // Normalize each row
                    for (int i = 0; i < rows; i++)
                    {
                        float[] row = matrix.GetRow(i);
                        float normValue = 0;

                        if (norm == "l1")
                        {
                            normValue = row.Sum(x => Math.Abs(x));
                        }
                        else if (norm == "l2")
                        {
                            normValue = (float)Math.Sqrt(row.Sum(x => x * x));
                        }
                        else if (norm == "max")
                        {
                            normValue = row.Max();
                        }

                        if (normValue > 0)
                        {
                            for (int j = 0; j < cols; j++)
                            {
                                result[i, j] = matrix[i, j] / normValue;
                            }
                        }
                    }
                }
                else
                {
                    // Normalize each column
                    for (int j = 0; j < cols; j++)
                    {
                        float[] col = matrix.GetColumn(j);
                        float normValue = 0;

                        if (norm == "l1")
                        {
                            normValue = col.Sum(x => Math.Abs(x));
                        }
                        else if (norm == "l2")
                        {
                            normValue = (float)Math.Sqrt(col.Sum(x => x * x));
                        }
                        else if (norm == "max")
                        {
                            normValue = col.Max();
                        }

                        if (normValue > 0)
                        {
                            for (int i = 0; i < rows; i++)
                            {
                                result[i, j] = matrix[i, j] / normValue;
                            }
                        }
                    }
                }

                return result;
            }
            else
            {
                throw new ArgumentException("Unsupported normalization type");
            }
        }
    }

    // Extension methods for array operations
    public static class ArrayExtensions
    {
        public static float[] GetRow(this float[,] matrix, int row)
        {
            int cols = matrix.GetLength(1);
            float[] result = new float[cols];
            for (int i = 0; i < cols; i++)
            {
                result[i] = matrix[row, i];
            }
            return result;
        }

        public static float[] GetColumn(this float[,] matrix, int col)
        {
            int rows = matrix.GetLength(0);
            float[] result = new float[rows];
            for (int i = 0; i < rows; i++)
            {
                result[i] = matrix[i, col];
            }
            return result;
        }

        public static float Max(this float[] array)
        {
            //return array.Max();
            return Enumerable.Max(array); 
        }
    }
}
