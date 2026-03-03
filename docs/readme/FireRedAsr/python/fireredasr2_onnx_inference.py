#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FireRedASR2-AED ONNX 推理：ONNX Encoder + ONNX Decoder + ONNX CTC
模型下载
git clone https://www.modelscope.cn/manyeyes/fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212.git
"""

import os
import math
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import logging
from typing import List, Tuple, Optional
import kaldiio
import kaldi_native_fbank as knf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ========== 配置路径 ==========
ONNX_DIR = "/path/to/fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212"
CMVN_FILE = os.path.join(ONNX_DIR, "cmvn.ark")
DICT_PATH = os.path.join(ONNX_DIR, "tokens.txt")
ENCODER_ONNX = os.path.join(ONNX_DIR, "encoder.int8.onnx")
DECODER_ONNX = os.path.join(ONNX_DIR, "decoder.int8.onnx")
CTC_ONNX = os.path.join(ONNX_DIR, "ctc.int8.onnx")

# 音频文件路径
WAV_PATH = "/path/to/test_wavs/0.wav"
# WAV_PATH = "/workspace/FireRedASR2S-main/assets/wav/TEST_MEETING_T0000000001_S00000.wav"
# WAV_PATH = "/workspace/FireRedASR2S-main/assets/wav/IT0011W0001.wav"
# WAV_PATH = "/workspace/FireRedASR2S-main/assets/wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav"
# WAV_PATH = "/workspace/FireRedASR2S-main/assets/wav/BAC009S0764W0121.wav"

# ========== 模型参数 ==========
d_model = 1280
n_layers_dec = 16
vocab_size = 8667
SOS_ID = 3
EOS_ID = 4
PAD_ID = 2
BLANK_ID = 0

# 下采样因子
SUBSAMPLING_FACTOR = 4
FRAME_SHIFT = 10  # ms
ENC_FRAME_SHIFT_SEC = FRAME_SHIFT / 1000.0 * SUBSAMPLING_FACTOR

# ======================== 集成 FeatExtractor 相关类 ========================
class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean * mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            logger.warning("Check data, len(feat) == 0")
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat


class FeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
                                    frame_shift=10, dither=0.0)

    def __call__(self, wav_paths, wav_uttids):
        feats = []
        durs = []
        return_wav_paths = []
        return_wav_uttids = []

        wav_datas = []
        if isinstance(wav_paths[0], str):
            for wav_path in wav_paths:
                sample_rate, wav_np = kaldiio.load_mat(wav_path)
                wav_datas.append([sample_rate, wav_np])
        else:
            wav_datas = wav_paths

        for (sample_rate, wav_np), path, uttid in zip(wav_datas, wav_paths, wav_uttids):
            dur = wav_np.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, wav_np))
            if fbank.shape[0] < 1:
                continue
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            fbank = torch.from_numpy(fbank).float()
            feats.append(fbank)
            durs.append(dur)
            return_wav_paths.append(path)
            return_wav_uttids.append(uttid)
        if len(feats) > 0:
            lengths = torch.tensor([feat.size(0) for feat in feats]).long()
            feats_pad = self.pad_feat(feats, 0.0)
        else:
            lengths, feats_pad = None, None
        return feats_pad, lengths, durs, return_wav_paths, return_wav_uttids

    def pad_feat(self, xs, pad_value):
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad

# ======================== 集成 TokenDict ========================
class TokenDict:
    """TokenDict"""

    def __init__(self, dict_path, unk="<unk>"):
        self.token2id = {}
        self.id2token = {}
        self.unk = unk
        self.unk_id = 0

        with open(dict_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                token = parts[0]
                # 兼容两种格式：一行是 token id 或只有 token
                if len(parts) >= 2 and parts[1].isdigit():
                    tid = int(parts[1])
                else:
                    tid = idx
                self.token2id[token] = tid
                self.id2token[tid] = token

        # 设置unk id
        if unk in self.token2id:
            self.unk_id = self.token2id[unk]
        else:
            self.token2id[unk] = self.unk_id
            self.id2token[self.unk_id] = unk

    def __getitem__(self, idx):
        """通过id获取token"""
        return self.id2token.get(idx, self.unk)

# ========== 加载 ONNX 模型 ==========
def load_onnx_model(path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, sess_options, providers=['CPUExecutionProvider'])
    logger.info(f"Loaded ONNX: {path}")
    return session

encoder_sess = load_onnx_model(ENCODER_ONNX)
decoder_sess = load_onnx_model(DECODER_ONNX)
ctc_sess = load_onnx_model(CTC_ONNX)

# 验证 decoder 输入数量
decoder_inputs = decoder_sess.get_inputs()
n_layers_onnx = len(decoder_inputs) - 3  # 减去 ys, encoder_outputs, src_mask
assert n_layers_onnx == n_layers_dec, f"Decoder layer mismatch: ONNX has {n_layers_onnx}, expected {n_layers_dec}"

# ========== 初始化特征提取器和词典 ==========
feat_extractor = FeatExtractor(kaldi_cmvn_file=CMVN_FILE)
token_dict = TokenDict(DICT_PATH, unk='<unk>')

# ========== 特征提取 ==========
def extract_features(wav_path):
    feats_pad, lengths, durs, _, _ = feat_extractor([wav_path], ['dummy_uttid'])
    return feats_pad, lengths

# ========== Encoder ONNX 推理 ==========
def run_encoder(feats: torch.Tensor, feat_lengths: torch.Tensor):
    feats_np = feats.numpy().astype(np.float32)
    feat_len_np = feat_lengths.numpy().astype(np.int64)
    inputs = {'input': feats_np, 'input_lengths': feat_len_np}
    outputs = encoder_sess.run(['output', 'output_lengths', 'mask'], inputs)
    enc_out, enc_len, enc_mask = outputs
    if enc_mask.dtype != np.bool_:
        enc_mask = enc_mask.astype(np.bool_)
    return torch.from_numpy(enc_out), torch.from_numpy(enc_len), torch.from_numpy(enc_mask)

# ========== CTC ONNX 推理 ==========
def run_ctc(enc_out: torch.Tensor):
    enc_out_np = enc_out.numpy().astype(np.float32)
    inputs = {'encoder_outputs': enc_out_np}
    logits = ctc_sess.run(['logits'], inputs)[0]
    return torch.from_numpy(logits)

# ========== 使用 ONNX Decoder 进行贪心解码 ==========
def greedy_decode_with_onnx(decoder_sess, enc_out: torch.Tensor, src_mask: torch.Tensor,
                            max_len_ratio: float = 1.0, sos_id=SOS_ID, eos_id=EOS_ID):
    """
    使用 ONNX Decoder 进行贪心解码，限制最大长度为 encoder帧数（即 max_len_ratio=1.0）
    """
    batch_size = enc_out.size(0)
    enc_time = enc_out.size(1)
    max_len = int(enc_time * max_len_ratio)
    logger.info(f"Decoder max_len set to {max_len} (enc_time={enc_time})")

    # 转换为 numpy
    enc_out_np = enc_out.numpy().astype(np.float32)
    src_mask_np = src_mask.numpy().astype(np.bool_)

    # 初始序列 ys = [SOS]
    ys = np.array([[sos_id]], dtype=np.int64)  # (1,1)

    # 初始化缓存：每层为空 (1, 0, d_model)
    caches = [np.empty((1, 0, d_model), dtype=np.float32) for _ in range(n_layers_dec)]

    token_list = []
    for step in range(max_len):
        # 准备输入字典
        input_dict = {
            'ys': ys,
            'encoder_outputs': enc_out_np,
            'src_mask': src_mask_np,
        }
        for i, cache in enumerate(caches):
            input_dict[f'cache_{i}'] = cache

        # 运行 ONNX
        output_names = ['logits'] + [f'new_cache_{i}' for i in range(n_layers_dec)]
        outputs = decoder_sess.run(output_names, input_dict)
        logits = outputs[0]  # (1, vocab_size)
        new_caches = outputs[1:]  # 每层的新缓存

        # 取 argmax 得到下一个 token
        next_token = int(np.argmax(logits, axis=-1).item())
        token_list.append(next_token)

        # 调试：打印 logits top5
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        top5 = np.argsort(probs[0])[-5:][::-1]
        logger.debug(f"Step {step}: token {next_token}, top5 ids: {top5}, probs: {probs[0][top5]}")

        # 更新序列和缓存
        ys = np.concatenate([ys, [[next_token]]], axis=1)
        caches = new_caches

        if next_token == eos_id:
            logger.info(f"Stopped at step {step} with EOS")
            break

    logger.info(f"Generated {len(token_list)} tokens (including possible EOS)")
    # 移除 SOS 和 EOS
    tokens = ys[0, 1:].tolist()  # 去掉开头的 SOS
    if tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    return tokens

# ========== 强制对齐获取时间戳 ==========
def get_ctc_timestamp(ctc_logits: torch.Tensor, tokens: List[int],
                      blank_id=BLANK_ID, frame_shift=ENC_FRAME_SHIFT_SEC):
    if len(tokens) == 0:
        return None, None
    log_probs = ctc_logits.log_softmax(dim=-1)  # (1, T, C)
    T = log_probs.size(1)
    if len(tokens) > T:
        logger.warning(f"Token length ({len(tokens)}) > log_probs length ({T}), cannot align")
        return None, None
    targets = torch.tensor([tokens], dtype=torch.int32)
    try:
        alignment, _ = torchaudio.functional.forced_align(log_probs, targets, blank=blank_id)
    except RuntimeError as e:
        logger.warning(f"Forced alignment failed: {e}")
        return None, None
    alignment = alignment[0].cpu().numpy()  # (T,)

    # 将 alignment 转换为每个 token 的起止时间
    start_times, end_times = [], []
    prev_token = blank_id
    for t, token in enumerate(alignment):
        if token != blank_id:
            if token != prev_token:  # 新 token 开始
                if prev_token != blank_id:
                    end_times.append(t * frame_shift)
                start_times.append(t * frame_shift)
                prev_token = token
        else:
            if prev_token != blank_id:
                end_times.append(t * frame_shift)
                prev_token = blank_id
    if prev_token != blank_id:
        end_times.append(len(alignment) * frame_shift)

    if len(start_times) != len(tokens):
        logger.warning(f"Align length mismatch: got {len(start_times)} segments, expected {len(tokens)}")
        return None, None
    return start_times, end_times

# ========== 主流程 ==========
def main():
    # 1. 提取特征
    feats, lengths = extract_features(WAV_PATH)
    logger.info(f"Features shape: {feats.shape}, length: {lengths.item()}")

    # 2. Encoder ONNX
    enc_out, enc_len, enc_mask = run_encoder(feats, lengths)
    logger.info(f"Encoder output: {enc_out.shape}, length: {enc_len.item()}")

    # 3. 使用 ONNX Decoder 贪心解码（限制长度与 encoder 帧数相同）
    tokens = greedy_decode_with_onnx(decoder_sess, enc_out, enc_mask, max_len_ratio=1.0)
    logger.info(f"Decoded token IDs: {tokens}")

    # 4. 转换为文本
    text = ''.join([token_dict[t] for t in tokens])
    print(f"\n识别文本: {text}")

    # 5. CTC 推理并获取时间戳
    ctc_logits = run_ctc(enc_out)
    start_times, end_times = get_ctc_timestamp(ctc_logits, tokens, blank_id=BLANK_ID)

    if start_times is not None:
        print("时间戳 (秒):")
        for i, (s, e) in enumerate(zip(start_times, end_times)):
            token_str = token_dict[tokens[i]]
            print(f"  {token_str}: {s:.3f} - {e:.3f}")
    else:
        print("无法生成时间戳")

if __name__ == "__main__":
    # 设置日志级别为 DEBUG 以查看每一步的 token 分布
    logger.setLevel(logging.DEBUG)
    main()