// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManySpeech.WenetAsr2.Model;
using KaldiNativeFbankSharp;
using System.Runtime.InteropServices;
using System.Data;
using System.Text.Json;
using YamlDotNet.Core;

namespace ManySpeech.WenetAsr2
{
    /// <summary>
    /// WavFrontend
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    internal class WavFrontend
    {
        private FrontendConfEntity _frontendConfEntity;
        OnlineFbank _onlineFbank;

        public WavFrontend(FrontendConfEntity frontendConfEntity)
        {
            _frontendConfEntity = frontendConfEntity;
            _onlineFbank = new OnlineFbank(
                dither: _frontendConfEntity.dither,
                snip_edges: _frontendConfEntity.snip_edges,
                sample_rate: _frontendConfEntity.fs,
                num_bins: _frontendConfEntity.n_mels,
                window_type:_frontendConfEntity.window
                );
        }

        public float[] GetFbank(float[] samples)
        {
            samples = samples.Select(x => (float)Float32ToInt16(x)).ToArray();
            float[] fbanks = _onlineFbank.GetFbank(samples);
            return fbanks;
        }
        public void InputFinished()
        {
            _onlineFbank.InputFinished();
        }
        private static Int16 Float32ToInt16(float sample)
        {
            if (sample < -0.999999f)
            {
                return Int16.MinValue;
            }
            else if (sample > 0.999999f)
            {
                return Int16.MaxValue;
            }
            else
            {
                if (sample < 0)
                {
                    return (Int16)(Math.Floor(sample * 32767.0f));
                }
                else
                {
                    return (Int16)(Math.Ceiling(sample * 32767.0f));
                }
            }
        }
    }
}
