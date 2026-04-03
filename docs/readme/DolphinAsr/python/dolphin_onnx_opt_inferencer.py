import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import subprocess
import time
import onnx

# ==================== 配置 ====================
SAMPLE_RATE = 16000
MAX_DECODE_STEPS = 1000
CPU_THREADS = 1

# 特征提取参数（与训练一致）
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
N_MELS = 80
FMIN = 0
FMAX = 8000


# =============================================

class TokenConverter:
    def __init__(self, token_file_path):
        self.token2id_dict = {}
        self.id2token_dict = {}
        with open(token_file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f.readlines() if line.strip()]
        for idx, token in enumerate(tokens):
            self.token2id_dict[token] = idx
            self.id2token_dict[idx] = token

    def ids2tokens(self, ids):
        return [self.id2token_dict.get(i, f"<unk:{i}>") for i in ids]

    def token2id(self, token):
        return self.token2id_dict.get(token)

def load_audio(file_path):
    """使用ffmpeg加载音频，返回float32波形"""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def extract_fbank(waveform):
    """从原始波形提取log-mel fbank特征 (T, n_mels)"""
    waveform = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0,
        normalized=False
    )
    mel_spec = mel_transform(waveform)
    log_mel = torch.log(mel_spec + 1e-6)
    return log_mel.squeeze().transpose(0, 1).numpy().astype(np.float32)

def preprocess_audio(audio_path):
    """加载音频并提取fbank特征，返回 (1, T, D) 和长度 (1,)"""
    waveform = load_audio(audio_path)
    feats = extract_fbank(waveform)           # (T, D)
    feats = feats[np.newaxis, :, :]           # (1, T, D)
    feats_len = np.array([feats.shape[1]], dtype=np.int64)
    return feats, feats_len

def get_input_names(model_path):
    """获取ONNX模型的输入名称"""
    model = onnx.load(model_path)
    return [inp.name for inp in model.graph.input]

class DolphinONNXInferencer:
    def __init__(self, encoder_path, decoder_path, tokens_path):
        self.converter = TokenConverter(tokens_path)
        self.sos_id = self.converter.token2id("<sos>")
        self.eos_id = self.converter.token2id("<eos>")
        self.asr_id = self.converter.token2id("<asr>")

        # 获取模型输入名称
        self.enc_input_names = get_input_names(encoder_path)
        self.dec_input_names = get_input_names(decoder_path)
        print(f"Encoder inputs: {self.enc_input_names}")
        print(f"Decoder inputs: {self.dec_input_names}")

        # 创建会话
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = CPU_THREADS
        self.encoder_session = ort.InferenceSession(encoder_path, opts)
        self.decoder_session = ort.InferenceSession(decoder_path, opts)

    def decode(self, speech_np, speech_lengths_np, lang_sym="zh", region_sym="CN"):
        """
        speech_np: (1, T, D) fbank特征
        speech_lengths_np: (1,) 特征长度
        """
        start_time = time.time()

        # 1. Encoder推理
        enc_inputs = {
            self.enc_input_names[0]: speech_np,
            self.enc_input_names[1]: speech_lengths_np
        }
        enc_out = self.encoder_session.run(None, enc_inputs)[0]
        enc_time = time.time() - start_time

        # 2. 构造初始token序列
        lang_id = self.converter.token2id(f"<{lang_sym}>")
        region_id = self.converter.token2id(f"<{region_sym}>")
        hyp = [self.sos_id, lang_id, region_id, self.asr_id]

        # 3. 贪婪解码
        dec_start = time.time()
        for step in range(MAX_DECODE_STEPS):
            dec_inputs = {
                self.dec_input_names[0]: enc_out,
                self.dec_input_names[1]: np.array([hyp], dtype=np.int64)
            }
            logits = self.decoder_session.run(None, dec_inputs)[0]
            next_id = int(np.argmax(logits[0]))
            hyp.append(next_id)
            if next_id == self.eos_id or len(hyp) >= MAX_DECODE_STEPS:
                break

        dec_time = time.time() - dec_start
        total_time = time.time() - start_time

        # 4. 转换结果
        tokens = self.converter.ids2tokens(hyp)
        full_text = "".join(tokens)
        clean_text = "".join(t for t in tokens if len(t) == 1 or (not t.startswith("<") and not t.endswith(">")))

        # 5. 计算RTF（基于原始音频时长，需要外部传入，此处用特征时长估算）
        audio_duration = speech_lengths_np[0] * HOP_LENGTH / SAMPLE_RATE  # 帧数 * 帧移 / 采样率
        rtf = total_time / audio_duration

        return {
            "text": full_text,
            "clean_text": clean_text,
            "rtf": round(rtf, 3),
            "steps": len(hyp) - 4,   # 减去初始的4个特殊token
            "enc_time": round(enc_time, 3),
            "dec_time": round(dec_time, 3),
            "total_time": round(total_time, 3)
        }

def main():
    # # 模型路径（请按实际情况修改）
    model_dir = "/to/path/DolphinAsr-base-int8-onnx"
    audio_path = f"{model_dir}/audio.wav"
    encoder_onnx_path = f"{model_dir}/encoder.int8.onnx"
    decoder_onnx_path = f"{model_dir}/decoder.int8.onnx"
    tokens_path = f"{model_dir}/tokens.txt"

    print("Loading audio and extracting features...")
    speech_np, speech_lengths_np = preprocess_audio(audio_path)

    print("Initializing inference engine...")
    inferencer = DolphinONNXInferencer(encoder_onnx_path, decoder_onnx_path, tokens_path)

    print("Starting inference...")
    result = inferencer.decode(speech_np, speech_lengths_np, lang_sym="zh", region_sym="CN")

    print("\n" + "=" * 80)
    print("Inference Results:")
    print("=" * 80)
    print(f"Full text: {result['text']}")
    print(f"Clean text: {result['clean_text']}")
    print(f"Steps: {result['steps']}")
    print(f"Total time: {result['total_time']}s, RTF: {result['rtf']}")
    print("=" * 80)

if __name__ == "__main__":
    main()