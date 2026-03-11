import time
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import warnings
from funasr.register import tables
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

warnings.filterwarnings('ignore')


class FunASRNanoONNX:
    def __init__(self, model_path, sample_rate=16000, n_mels=80,
                 window_length=400, hop_length=160, pre_emphasize=0.97,
                 lfr_m=7, lfr_n=6, stop_tokens=None,
                 max_new_tokens=200, repeat_penalty=1.0, max_audio_len=512000,
                 num_threads=4):
        self.sample_rate = sample_rate
        self.max_new_tokens = max_new_tokens
        self.repeat_penalty = repeat_penalty
        self.max_audio_len = max_audio_len
        self.stop_tokens = stop_tokens if stop_tokens else [151643, 151645]

        # ONNX Runtime 会话配置
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        providers = ['CPUExecutionProvider']

        self.ort_encoder = onnxruntime.InferenceSession(
            f'{model_path}/encoder_adaptor.int8.onnx', opts, providers=providers)
        self.ort_embed = onnxruntime.InferenceSession(
            f'{model_path}/embed.int8.onnx', opts, providers=providers)
        self.ort_decoder = onnxruntime.InferenceSession(
            f'{model_path}/decoder.int8.onnx', opts, providers=providers)

        # 解析 decoder 结构
        decoder_inputs = self.ort_decoder.get_inputs()
        self.num_layers = sum(1 for inp in decoder_inputs if 'in_key' in inp.name)
        offset = self.num_layers * 2
        self.idx_hidden = offset
        self.vocab_size = self.ort_decoder.get_outputs()[self.idx_hidden].shape[-1]
        sample_key = decoder_inputs[0]
        self.num_heads = sample_key.shape[1]
        self.head_dim = sample_key.shape[3]
        self.embed_in_name = self.ort_embed.get_inputs()[0].name
        self.embed_out_name = self.ort_embed.get_outputs()[0].name

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 初始化 FunASR 前端（包含 LFR）
        frontend_conf = {
            "fs": sample_rate,
            "window": "hamming",
            "n_mels": n_mels,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": lfr_m,
            "lfr_n": lfr_n,
            "dither": 1.0,
            "snip_edges": True,
            "cmvn_file": None,
        }
        frontend_class = tables.frontend_classes.get("wav_frontend")
        self.frontend = frontend_class(**frontend_conf)

        # 预计算系统提示嵌入
        sys_ids = self.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                                         return_tensors='np').astype(np.int64)
        user_ids = self.tokenizer.encode("<|im_start|>user\n", return_tensors='np').astype(np.int64)
        asst_ids = self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n",
                                          return_tensors='np').astype(np.int64)
        self.system_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: sys_ids})[0]
        self.user_prefix_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: user_ids})[0]
        self.assistant_prefix_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: asst_ids})[0]

    def _load_and_process_audio(self, audio_path: str):
        """加载音频并提取 fbank 特征"""
        try:
            # 加载音频（自动重采样至 self.sample_rate）
            data_src = load_audio_text_image_video(audio_path, fs=self.sample_rate)
            if len(data_src) > self.max_audio_len:
                data_src = data_src[:self.max_audio_len]
            audio_duration = len(data_src) / self.sample_rate

            # 提取 fbank 特征
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type="sound",
                frontend=self.frontend,
                is_final=True,
            )

            # 转换为 numpy 数组
            speech_np = speech.cpu().numpy().astype(np.float32)
            speech_lengths_np = speech_lengths.cpu().numpy().astype(np.int64)

            return speech_np, speech_lengths_np, audio_duration

        except Exception as e:
            raise RuntimeError(f"音频处理失败: {e}")

    def _encode_prompt(self, prompt):
        ids = self.tokenizer.encode(prompt, return_tensors='np').astype(np.int64)
        return self.ort_embed.run([self.embed_out_name], {self.embed_in_name: ids})[0]

    def _encode_audio(self, audio_path):
        feat, feat_len, audio_duration = self._load_and_process_audio(audio_path)
        audio_embed = self.ort_encoder.run(
            ['encoded_features'],
            {'speech': feat, 'speech_lengths': feat_len}
        )[0]
        return audio_embed, audio_duration

    def _decode(self, init_hidden, init_len):
        batch_size = 1
        inputs = {}
        for i in range(self.num_layers):
            inputs[f'in_key_{i}'] = np.zeros((batch_size, self.num_heads, 1, self.head_dim, 0), dtype=np.float32)
            inputs[f'in_value_{i}'] = np.zeros((batch_size, self.num_heads, 1, 0, self.head_dim), dtype=np.float32)

        inputs['hidden_states'] = init_hidden.astype(np.float32)
        inputs['history_len'] = np.array([0], dtype=np.int64)
        inputs['ids_len'] = np.array([init_len], dtype=np.int64)
        inputs['attention_mask'] = np.array([1], dtype=np.int8)

        generated = []
        penalty = np.ones((1, self.vocab_size), dtype=np.float32)
        step = 0

        while step < self.max_new_tokens:
            outputs = self.ort_decoder.run(None, inputs)
            logits = outputs[self.idx_hidden]

            penalized = logits * penalty
            next_token = np.argmax(penalized, axis=-1, keepdims=True).astype(np.int64)
            token_id = int(next_token[0, 0])
            if token_id in self.stop_tokens:
                break
            generated.append(token_id)
            penalty[0, token_id] *= self.repeat_penalty

            for i in range(self.num_layers):
                inputs[f'in_key_{i}'] = outputs[i]
                inputs[f'in_value_{i}'] = outputs[i + self.num_layers]

            token_embed = self.ort_embed.run(
                [self.embed_out_name],
                {self.embed_in_name: next_token.astype(np.int64)}
            )[0]
            inputs['hidden_states'] = token_embed

            kv_len = outputs[-1]
            if isinstance(kv_len, np.ndarray):
                kv_len = kv_len[0]
            inputs['history_len'] = np.array([kv_len], dtype=np.int64)
            inputs['ids_len'] = np.array([1], dtype=np.int64)
            inputs['attention_mask'] = np.array([0], dtype=np.int8)
            step += 1

        return generated

    def transcribe(self, audio_path, prompt):
        audio_embed, audio_duration = self._encode_audio(audio_path)
        prompt_embed = self._encode_prompt(prompt)

        concat = np.concatenate([
            self.system_embed,
            self.user_prefix_embed,
            prompt_embed,
            audio_embed,
            self.assistant_prefix_embed
        ], axis=1)

        start = time.time()
        token_ids = self._decode(concat, concat.shape[1])
        elapsed = time.time() - start

        if token_ids:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        else:
            text = ""
        return text, len(token_ids), elapsed, audio_duration


if __name__ == "__main__":
    # 请将以下路径替换为实际的模型路径
    model_path = '/path/to/Fun-ASR-Nano-2512-LLM-int8-onnx'

    test_audio = [
        '/path/to/example/zh_31s.wav',
        '/path/to/example/zh.mp3',
        '/path/to/example/en.mp3',
        '/path/to/example/ja.mp3',
        '/path/to/example/yue.mp3'
    ]
    hotwords = ['开放时间']
    hotwords_str = ", ".join(hotwords) if hotwords else ""
    languages = ["中文", "中文", "英文", "日语", "粤语"]

    itn = True
    prompts = []
    for lang in languages:
        base = ("请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
                "**上下文信息：**\n\n\n")
        if hotwords_str:
            base += f"热词列表：[{hotwords_str}]\n"
        base += f"语音转写成{lang}"
        if not itn:
            base += "，不进行文本规整"
        base += "："
        prompts.append(base)

    asr = FunASRNanoONNX(
        model_path,
        max_audio_len=400000, # 25秒
        num_threads=8,
    )

    for idx, (path, prompt) in enumerate(zip(test_audio, prompts)):
        print(f"\n--- 测试 {idx+1}: {path} ---")
        text, num_tokens, elapsed, audio_len = asr.transcribe(path, prompt)
        if text:
            print(f"音频时长: {audio_len:.2f}s")
            print(f"转写结果: {text}")
            print(f"生成 {num_tokens} tokens, 耗时 {elapsed:.2f}s, RTF={elapsed/audio_len:.3f}")
        else:
            print("未生成任何文本")