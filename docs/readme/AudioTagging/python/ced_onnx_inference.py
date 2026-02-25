import torch
import torchaudio
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

# 音频预处理配置
SR = 16000  # 采样率
N_MELS = 64  # mel频谱维度
HOP_LENGTH = 160  # 帧移
TARGET_LENGTH = 1012  # 模型期望的时间帧数
CHUNK_LEN = 10.0  # 音频chunk长度（秒）

# 文件路径配置
LABEL_TOKENS = "/path/to/tokens.txt"
ONNX_MODEL_PATH = "/path/to/model.onnx"
AUDIO_PATH = "/path/to/1.wav"
TOP_K = 3  # 输出top-k预测结果

# ===================== 标签映射加载（TokenConverter类） =====================
class TokenConverter:
    def __init__(self, token_file_path):
        self.token2id_dict = {}
        self.id2token_dict = {}

        # 读取tokens文件并构建映射
        with open(token_file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f.readlines() if line.strip()]

        for idx, token in enumerate(tokens):
            self.token2id_dict[token] = idx
            self.id2token_dict[idx] = token

    def ids2tokens(self, ids):
        """将ID列表转换为对应的标签列表"""
        return [self.id2token_dict.get(token_id, f"<unk:{token_id}>") for token_id in ids]

    def token2id(self, token):
        """将标签转换为对应的ID"""
        return self.token2id_dict.get(token)

# ===================== 音频预处理 =====================
def preprocess_audio_to_mel(waveform, sr):
    """
    将音频波形转换为模型所需的 Mel 频谱
    Args:
        waveform: 音频波形，shape [time] 或 [batch, time]
        sr: 采样率
    Returns:
        mel_spec: Mel频谱，shape [1, n_mels, target_length]
    """
    # 统一采样率
    if sr != SR:
        waveform = torchaudio.transforms.Resample(sr, SR)(waveform)

    # 保证是2维（batch, time）
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Mel频谱转换
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=512,
        hop_length=HOP_LENGTH,
        win_length=512,
        n_mels=N_MELS,
        f_min=0,
        f_max=8000,
        power=2.0
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)

    mel = mel_transform(waveform)  # [1, n_mels, time]
    mel_db = amp_to_db(mel)  # 转换为dB

    # 调整时间维度到目标长度（截断/填充）
    if mel_db.size(-1) > TARGET_LENGTH:
        mel_db = mel_db[..., :TARGET_LENGTH]
    elif mel_db.size(-1) < TARGET_LENGTH:
        pad = torch.zeros(mel_db.size(0), N_MELS, TARGET_LENGTH - mel_db.size(-1))
        mel_db = torch.cat([mel_db, pad], dim=-1)

    return mel_db

# ===================== ONNX模型推理 =====================
def onnx_inference(onnx_model_path, audio_path, token_converter, top_k=3):
    """
    ONNX模型推理主函数（适配TokenConverter）
    Args:
        onnx_model_path: ONNX模型文件路径
        audio_path: 待推理的音频文件路径
        token_converter: TokenConverter实例
        top_k: 输出前k个预测结果
    Returns:
        topk_results: 包含(标签名称, 概率)的列表
    """
    # 1. 加载音频文件
    waveform, sr = torchaudio.load(audio_path)
    print(f"Loaded audio: {audio_path}, sample rate: {sr}, shape: {waveform.shape}")

    # 2. 预处理为Mel频谱
    mel_input = preprocess_audio_to_mel(waveform, sr)
    print(f"Preprocessed mel spec shape: {mel_input.shape}")

    # 3. 初始化ONNX Runtime会话
    ort_session = ort.InferenceSession(
        onnx_model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用GPU
    )

    # 4. 执行推理
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: mel_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]  # [1, num_classes]

    # 5. 解析top-k结果（使用TokenConverter映射ID到标签）
    logits = ort_output[0]
    topk_indices = np.argsort(logits)[::-1][:top_k]
    topk_probs = logits[topk_indices]
    topk_tokens = token_converter.ids2tokens(topk_indices)  # 核心修改：用TokenConverter转换ID

    # 6. 格式化输出结果
    topk_results = []
    print("\n=== Top-k Prediction Results ===")
    for k, (token, prob) in enumerate(zip(topk_tokens, topk_probs)):
        topk_results.append((token, prob))
        print(f"Top{k + 1}: {token:<30} Probability: {prob:.4f}")

    return topk_results

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 加载TokenConverter
    token_converter = TokenConverter(LABEL_TOKENS)
    print(f"Loaded token file: {LABEL_TOKENS}, total tokens: {len(token_converter.id2token_dict)}")

    # 执行ONNX推理
    results = onnx_inference(ONNX_MODEL_PATH, AUDIO_PATH, token_converter, TOP_K)