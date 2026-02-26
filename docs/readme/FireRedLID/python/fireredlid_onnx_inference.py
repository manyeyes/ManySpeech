#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import torch
import onnxruntime as ort
import logging
import kaldiio
import kaldi_native_fbank as knf

# ======================== 配置参数 ========================
WAV_PATH = "/path/to/hello_zh.wav"
MODEL_DIR = "/path/to/FireRedLID-int8-onnx"
ENCODER_ONNX = os.path.join(MODEL_DIR, "encoder.int8.onnx")
DECODER_ONNX = os.path.join(MODEL_DIR, "decoder.int8.onnx")
USE_GPU = False
BEAM_SIZE = 3
# =========================================================

# 基础配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


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


# ======================== 集成 TokenDict 和 LidTokenizer ========================
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


class LidTokenizer:
    def __init__(self, dict_path, unk="<unk>"):
        self.dict = TokenDict(dict_path, unk=unk)
        self.unk = unk
        # 从字典文件构建 token 到 id 的映射
        self.token2id = self.dict.token2id
        self.unk_id = self.dict.unk_id

    def detokenize(self, inputs, join_symbol=" "):
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)
        return s

    def token_to_id(self, token):
        """将 token 转换为对应的 id，若不存在则返回 <unk> 的 id"""
        return self.token2id.get(token, self.unk_id)

    def tokens_to_ids(self, tokens):
        """批量转换 token 列表为 id 列表"""
        return [self.token_to_id(t) for t in tokens]


# ======================== 核心推理逻辑 ========================
def init_components():
    """初始化所有组件（特征提取器/Tokenizer/ONNX会话/特殊token）"""
    # 加载特征提取器
    cmvn_path = os.path.join(MODEL_DIR, "cmvn.ark")
    if not os.path.exists(cmvn_path):
        raise FileNotFoundError(f"找不到CMVN文件: {cmvn_path}")
    feat_extractor = FeatExtractor(cmvn_path)

    # 加载Tokenizer和字典映射
    dict_path = os.path.join(MODEL_DIR, "dict.txt")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"找不到字典文件: {dict_path}")
    tokenizer = LidTokenizer(dict_path)

    # 解析特殊token
    sos_id = tokenizer.token_to_id('<sos>')
    eos_id = tokenizer.token_to_id('<eos>')
    pad_id = tokenizer.token_to_id('<pad>')
    token2id = tokenizer.token2id
    id2token = tokenizer.dict.id2token

    # 加载ONNX会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
    if not os.path.exists(ENCODER_ONNX) or not os.path.exists(DECODER_ONNX):
        raise FileNotFoundError("ONNX模型文件不存在")
    enc_session = ort.InferenceSession(ENCODER_ONNX, providers=providers)
    dec_session = ort.InferenceSession(DECODER_ONNX, providers=providers)

    return feat_extractor, tokenizer, sos_id, eos_id, pad_id, token2id, id2token, enc_session, dec_session


def process_audio(feat_extractor, enc_session):
    """提取音频特征并运行编码器"""
    # 特征提取
    if not os.path.exists(WAV_PATH):
        raise FileNotFoundError(f"音频文件不存在: {WAV_PATH}")
    uttid = os.path.splitext(os.path.basename(WAV_PATH))[0]
    try:
        feats, lengths, durs, _, _ = feat_extractor([WAV_PATH], [uttid])
    except Exception as e1:
        logger.warning(f"特征提取顺序1失败: {e1}")
        try:
            feats, lengths, durs, _, _ = feat_extractor([uttid], [WAV_PATH])
        except Exception as e2:
            raise RuntimeError(f"特征提取失败: {e2}") from e2

    # 特征维度适配
    expected_feat_dim = None
    for inp in enc_session.get_inputs():
        if inp.name == "features" and len(inp.shape) >= 3 and isinstance(inp.shape[2], int):
            expected_feat_dim = inp.shape[2]
            break
    if expected_feat_dim is None:
        raise RuntimeError("无法确定模型期望的特征维度")

    actual_feat_dim = feats.shape[2]
    if actual_feat_dim > expected_feat_dim:
        logger.warning(f"裁剪特征维度: {actual_feat_dim} -> {expected_feat_dim}")
        feats = feats[:, :, :expected_feat_dim]
    elif actual_feat_dim < expected_feat_dim:
        raise ValueError(f"特征维度不足: 实际{actual_feat_dim} < 期望{expected_feat_dim}")

    # 编码器推理
    feats_np = feats.numpy().astype(np.float32)
    lengths_np = lengths.numpy().astype(np.int64)
    output_names = [o.name for o in enc_session.get_outputs()]
    inputs = {"features": feats_np, "lengths": lengths_np}
    outputs = enc_session.run(output_names, inputs)

    # 正确校验输出
    enc_out, enc_mask = None, None
    for name, val in zip(output_names, outputs):
        if name == "encoder_out":
            enc_out = val
        elif name == "encoder_mask":
            enc_mask = val

    if enc_out is None:
        raise ValueError("未找到encoder_out输出")
    if enc_mask is None:
        raise ValueError("未找到encoder_mask输出")

    return enc_out, enc_mask


def beam_search_decode(dec_session, enc_out, enc_mask, sos_id, eos_id, id2token):
    """核心束搜索解码逻辑（完整保留两步束搜索）"""
    batch = 1
    d_model = enc_out.shape[2]
    n_layers = sum(1 for inp in dec_session.get_inputs() if inp.name.startswith("cache_"))

    # 第一步：初始sos
    ys = np.array([[sos_id]], dtype=np.int64)
    caches = [np.empty((batch, 0, d_model), dtype=np.float32) for _ in range(n_layers)]

    # 运行解码器第一步
    input_names = [i.name for i in dec_session.get_inputs()]
    inputs = {}
    for name in input_names:
        if name == "encoder_out":
            inputs[name] = enc_out
        elif name == "encoder_mask":
            inputs[name] = enc_mask.astype(np.bool_)
        elif name == "ys":
            inputs[name] = ys
        elif name.startswith("cache_"):
            inputs[name] = caches[int(name.split('_')[1])]
    output_names = [o.name for o in dec_session.get_outputs()]
    outputs = dec_session.run(output_names, inputs)
    logits1, caches1 = outputs[0], outputs[1:]

    # 计算第一步概率
    log_probs1 = logits1[0] - np.log(np.sum(np.exp(logits1[0])))
    top_indices1 = np.argsort(log_probs1)[::-1][:BEAM_SIZE]
    top_log_probs1 = log_probs1[top_indices1]

    # 遍历第一步候选，运行第二步
    candidates = []
    for i, token1 in enumerate(top_indices1):
        score = top_log_probs1[i]
        ys2 = np.array([[token1]], dtype=np.int64)

        # 深拷贝缓存
        import copy
        caches_copy = copy.deepcopy(caches1)

        # 第二步推理
        inputs["ys"] = ys2
        for name in input_names:
            if name.startswith("cache_"):
                inputs[name] = caches_copy[int(name.split('_')[1])]
        outputs2 = dec_session.run(output_names, inputs)
        logits2 = outputs2[0]

        log_probs2 = logits2[0] - np.log(np.sum(np.exp(logits2[0])))
        top_indices2 = np.argsort(log_probs2)[::-1][:BEAM_SIZE]
        for j, token2 in enumerate(top_indices2):
            new_score = score + log_probs2[token2]
            candidates.append((token1, token2, new_score, log_probs2[token2]))

    # 选择最优结果
    best_token1, best_token2, best_score, best_logprob2 = max(candidates, key=lambda x: x[2])
    final_token = best_token1
    confidence = np.exp(best_score / 2)

    # 解析结果
    token_id_py = int(final_token)
    raw_token = id2token.get(token_id_py, f"<UNK>{token_id_py}")
    clean_token = raw_token.split('\t')[0].strip() if '\t' in raw_token else raw_token.strip()

    # 打印关键信息
    logger.info(f"\n========== 推理结果 ==========")
    logger.info(f"音频文件: {WAV_PATH}")
    logger.info(f"预测语言: {clean_token}")
    logger.info(f"Token ID: {token_id_py}")
    logger.info(f"置信度: {confidence:.4f}")
    logger.info("==================================")


def main():
    try:
        # 初始化组件
        feat_extractor, tokenizer, sos_id, eos_id, pad_id, token2id, id2token, enc_session, dec_session = init_components()

        # 音频处理+编码器推理
        enc_out, enc_mask = process_audio(feat_extractor, enc_session)

        # 束搜索解码
        beam_search_decode(dec_session, enc_out, enc_mask, sos_id, eos_id, id2token)
    except Exception as e:
        logger.error(f"推理出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()