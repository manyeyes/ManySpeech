#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 ONNX 推理脚本 for FireRedVad
依赖：
    numpy, onnxruntime, soundfile, kaldi_native_fbank, kaldiio
安装：
    pip install numpy onnxruntime soundfile kaldi-native-fbank kaldiio
"""

import os
import math
import soundfile as sf
import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

# -------------------- 配置类 --------------------
class FireRedVadConfig:
    """VAD 参数配置"""
    def __init__(self,
                 use_gpu=False,
                 smooth_window_size=5,
                 speech_threshold=0.4,
                 min_speech_frame=20,
                 max_speech_frame=2000,
                 min_silence_frame=20,
                 merge_silence_frame=0,
                 extend_speech_frame=0,
                 chunk_max_frame=30000):
        self.use_gpu = use_gpu
        self.smooth_window_size = smooth_window_size
        self.speech_threshold = speech_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame
        self.chunk_max_frame = chunk_max_frame
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")

# -------------------- 特征提取 --------------------
class CMVN:
    """读取Kaldi CMVN文件并应用归一化"""
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variances = \
            self._read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        return (x - self.means) * self.inverse_std_variances

    def _read_kaldi_cmvn(self, kaldi_cmvn_file):
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = stats[0, :dim] / count
        variance = stats[1, :dim] / count - means * means
        variance = np.maximum(variance, floor)
        inverse_std = 1.0 / np.sqrt(variance)
        return dim, means, inverse_std


class KaldifeatFbank:
    """使用kaldi-native-fbank提取FBank特征"""
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.dither = dither
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if isinstance(wav, str):
            wav_np, sample_rate = sf.read(wav, dtype="int16")
        elif isinstance(wav, (tuple, list)) and len(wav) == 2:
            sample_rate, wav_np = wav
        else:
            raise TypeError("wav must be file path or (sample_rate, waveform) tuple")
        assert len(wav_np.shape) == 1

        self.opts.frame_opts.dither = self.dither if is_train else 0.0
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(sample_rate, wav_np.tolist())
        num_frames = fbank.num_frames_ready
        if num_frames == 0:
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack([fbank.get_frame(i) for i in range(num_frames)])
        return feat


class AudioFeat:
    """音频特征提取器：FBank + CMVN（返回 numpy 数组）"""
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
                                     frame_shift=10, dither=0)

    def extract(self, audio):
        if isinstance(audio, str):
            wav_np, sample_rate = sf.read(audio, dtype="int16")
        elif isinstance(audio, (list, tuple)):
            wav_np, sample_rate = audio
        else:
            wav_np = audio
            sample_rate = 16000
        assert sample_rate == 16000

        dur = wav_np.shape[0] / sample_rate
        fbank = self.fbank((sample_rate, wav_np))
        if self.cmvn is not None:
            fbank = self.cmvn(fbank)
        # 返回 numpy 数组，类型为 float32
        return fbank.astype(np.float32), dur

# -------------------- 后处理 --------------------
class VadPostprocessor:
    """VAD后处理：平滑、阈值、合并语音段"""
    def __init__(self, smooth_window_size, speech_threshold,
                 min_speech_frame, max_speech_frame,
                 min_silence_frame, merge_silence_frame,
                 extend_speech_frame):
        self.smooth_window_size = smooth_window_size
        self.speech_threshold = speech_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame

    def process(self, probs):
        """输入一维概率数组，返回平滑后的0/1决策列表"""
        probs = np.asarray(probs).flatten()
        # 平滑
        kernel = np.ones(self.smooth_window_size) / self.smooth_window_size
        smoothed = np.convolve(probs, kernel, mode='same')
        # 阈值
        decisions = (smoothed > self.speech_threshold).astype(int).tolist()
        # 合并短时语音/静音段
        return self._merge_short_segments(decisions)

    def _merge_short_segments(self, decisions):
        """合并过短的语音/静音段"""
        # 提取语音段
        segments = []
        start = None
        for i, d in enumerate(decisions):
            if d == 1 and start is None:
                start = i
            elif d == 0 and start is not None:
                segments.append((start, i-1))
                start = None
        if start is not None:
            segments.append((start, len(decisions)-1))

        # 过滤短语音并截断超长语音
        merged = []
        for s, e in segments:
            length = e - s + 1
            if length < self.min_speech_frame:
                continue
            if length > self.max_speech_frame:
                e = s + self.max_speech_frame - 1
            merged.append((s, e))

        # 合并短静音间隔
        if len(merged) > 1:
            final = [merged[0]]
            for i in range(1, len(merged)):
                gap = merged[i][0] - final[-1][1] - 1
                if gap <= self.merge_silence_frame:
                    final[-1] = (final[-1][0], merged[i][1])
                else:
                    final.append(merged[i])
            merged = final

        # 扩展语音段前后
        if self.extend_speech_frame > 0:
            extended = []
            for s, e in merged:
                s = max(0, s - self.extend_speech_frame)
                e = min(len(decisions)-1, e + self.extend_speech_frame)
                extended.append((s, e))
            merged = extended

        # 转换为决策列表
        new_decisions = [0] * len(decisions)
        for s, e in merged:
            for i in range(s, e+1):
                new_decisions[i] = 1
        return new_decisions

    def decision_to_segment(self, decisions, dur):
        """帧级决策转时间戳（秒），帧移10ms"""
        segments = []
        start = None
        for i, d in enumerate(decisions):
            if d == 1 and start is None:
                start = i
            elif d == 0 and start is not None:
                segments.append([start * 0.01, i * 0.01])  # 结束时间为下一帧起点
                start = None
        if start is not None:
            segments.append([start * 0.01, len(decisions) * 0.01])
        return segments

# -------------------- ONNX 推理封装 --------------------
class ONNXFireRedVad:
    def __init__(self, onnx_path, config, cmvn_path):
        self.config = config
        self.audio_feat = AudioFeat(cmvn_path)
        self.vad_postprocessor = VadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
            config.merge_silence_frame,
            config.extend_speech_frame,
        )

        # 创建ONNX推理会话
        providers = ['CPUExecutionProvider']
        if config.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

        # 解析输入输出信息
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]
        self.R = len([name for name in self.input_names if name.startswith('cache_')])

        # 获取cache形状（可能包含动态维度，如 (1,128,'C')）
        cache_input = self.sess.get_inputs()[1]  # 第一个输入是feat
        raw_shape = cache_input.shape

        # 根据已知模型参数固定cache形状（从导出脚本获知：P=128, lookback_padding=19）
        KNOWN_P = 128
        KNOWN_C = 19   # 必须与模型实际lookback_padding一致

        if any(isinstance(dim, str) for dim in raw_shape):
            print(f"检测到动态维度: {raw_shape}，使用已知固定值: (1, {KNOWN_P}, {KNOWN_C})")
            self.cache_shape = (1, KNOWN_P, KNOWN_C)
        else:
            self.cache_shape = raw_shape

        print(f"模型信息: {self.R} 个 cache, 形状 {self.cache_shape}")

    def detect(self, audio, do_postprocess=True):
        feats, dur = self.audio_feat.extract(audio)  # (T, 80) numpy数组
        T = feats.shape[0]
        print(f"特征形状: {feats.shape}, 音频时长: {dur:.3f}s")

        if T == 0:
            return {"dur": round(dur, 3), "timestamps": []}, np.array([])

        # 分块处理
        chunk_size = self.config.chunk_max_frame
        all_probs = []
        caches = [np.zeros(self.cache_shape, dtype=np.float32) for _ in range(self.R)]

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_feat = feats[start:end, :]  # (T_chunk, 80)
            # 增加batch维度 -> (1, T_chunk, 80)
            chunk_feat = np.expand_dims(chunk_feat, axis=0).astype(np.float32)

            feed_dict = {'feat': chunk_feat}
            for i, cache in enumerate(caches):
                feed_dict[f'cache_{i}'] = cache

            outputs = self.sess.run(self.output_names, feed_dict)
            probs_chunk = outputs[0].flatten()          # (T_chunk,)
            for i in range(self.R):
                caches[i] = outputs[1 + i]               # 更新缓存

            all_probs.append(probs_chunk)

        probs = np.concatenate(all_probs)                # (T,)
        print(f"概率统计: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

        if not do_postprocess:
            return None, probs

        # 简单阈值统计（可选）
        over_thresh = (probs > self.config.speech_threshold).sum()
        print(f"超过阈值的帧数: {over_thresh} / {len(probs)}")

        decisions = self.vad_postprocessor.process(probs)
        starts_ends_s = self.vad_postprocessor.decision_to_segment(decisions, dur)

        result = {"dur": round(dur, 3), "timestamps": starts_ends_s}
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, probs

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 路径配置（请根据实际情况修改）
    onnx_path = "/path/to/FireRedVad-onnx/model.onnx"
    cmvn_path = "/path/to/FireRedVad-onnx/cmvn.ark"
    audio_file = "/path/to/FireRedVad-onnx/0.wav"
    # VAD参数配置（建议根据实际需求调整）
    config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=20,       # 合并短静音（0.2秒）
        extend_speech_frame=20,        # 语音段前后扩展0.2秒
        chunk_max_frame=30000
    )

    vad = ONNXFireRedVad(onnx_path, config, cmvn_path)
    result, probs = vad.detect(audio_file)

    print(f"音频时长: {result['dur']} 秒")
    print("检测到的语音段:")
    for seg in result['timestamps']:
        print(f"  {seg[0]:.3f} - {seg[1]:.3f}")