import numpy as np
import onnxruntime as ort
import time
import onnx

# 配置
SAMPLE_RATE = 16000  
SPEECH_LENGTH = 30
MAX_DECODE_STEPS = 1000
CPU_THREADS = 1

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
        return [self.id2token_dict.get(token_id, f"<unk:{token_id}>") for token_id in ids]

    def token2id(self, token):
        return self.token2id_dict.get(token)

# 音频预处理
def load_audio(file_path):
    import subprocess
    import numpy as np

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-"
    ]

    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def preprocess_audio(audio_path):
    raw_waveform = load_audio(audio_path)
    target_length = int(SAMPLE_RATE * SPEECH_LENGTH)

    if len(raw_waveform) >= target_length:
        speech = raw_waveform[:target_length]
    else:
        speech = np.pad(raw_waveform, (0, target_length - len(raw_waveform)))

    return speech.reshape(1, -1), np.array([speech.shape[0]], dtype=np.int64)

# onnx推理
class DolphinONNXInferencer:
    def __init__(self, encoder_path, decoder_path, tokens_path):
        self.converter = TokenConverter(tokens_path)
        self.sos_id = 39999
        self.eos_id = 40000

        # 加载模型并获取输入名
        self.encoder_input_names = self._get_onnx_input_names(encoder_path)
        self.decoder_input_names = self._get_onnx_input_names(decoder_path)

        # 创建ONNX会话
        self.encoder_session = self._create_session(encoder_path)
        self.decoder_session = self._create_session(decoder_path)

    def _get_onnx_input_names(self, model_path):
        return [inp.name for inp in onnx.load(model_path).graph.input]

    def _create_session(self, model_path):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = CPU_THREADS
        return ort.InferenceSession(model_path, sess_options=sess_options)

    def decode(self, speech_np, speech_lengths_np, lang_sym="zh", region_sym="CN"):
        # Encoder推理
        enc_start = time.time()
        encoder_inputs = {
            self.encoder_input_names[0]: speech_np,
            self.encoder_input_names[1]: speech_lengths_np
        }
        enc_out = self.encoder_session.run(None, encoder_inputs)[0]
        enc_time = time.time() - enc_start

        # Decoder初始化解码
        hyp = np.zeros((1, MAX_DECODE_STEPS), dtype=np.int64)
        hyp[0, 0] = self.sos_id
        hyp[0, 1] = self.converter.token2id(f"<{lang_sym}>")
        hyp[0, 2] = self.converter.token2id(f"<{region_sym}>")
        hyp[0, 3] = self.converter.token2id("<asr>")

        current_len = 4
        dec_start = time.time()

        # 贪婪解码
        for step in range(MAX_DECODE_STEPS):
            decoder_inputs = {
                self.decoder_input_names[0]: enc_out,
                self.decoder_input_names[1]: hyp[:, :current_len]
            }

            logits = self.decoder_session.run(None, decoder_inputs)[0]
            next_id = int(np.argmax(logits[0]))

            if current_len >= MAX_DECODE_STEPS:
                break

            hyp[0, current_len] = next_id
            current_len += 1

            if next_id == self.eos_id:
                break

        dec_time = time.time() - dec_start
        total_time = enc_time + dec_time

        # 处理结果
        tokens = self.converter.ids2tokens(hyp[0, :current_len].tolist())
        full_text = "".join(tokens)
        clean_text = "".join([t for t in tokens if len(t) == 1 or (not t.startswith("<") and not t.endswith(">"))])

        audio_duration = speech_np.shape[1] / SAMPLE_RATE
        rtf = total_time / audio_duration

        return {
            "text": full_text,
            "clean_text": clean_text,
            "rtf": round(rtf, 3),
            "steps": step + 1,
            "enc_time": round(enc_time, 3),
            "dec_time": round(dec_time, 3),
            "total_time": round(total_time, 3)
        }


def main():
    model_dir = ""# "/to/path/DolphinAsr-base-int8-onnx/"
    audio_path = f"{model_dir}audio.wav"
    encoder_onnx_path = f"{model_dir}encoder.onnx"
    decoder_onnx_path = f"{model_dir}decoder.onnx"
    tokens_path = f"{model_dir}tokens.txt"

    print("Loading audio...")
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

    return result


if __name__ == "__main__":
    main()