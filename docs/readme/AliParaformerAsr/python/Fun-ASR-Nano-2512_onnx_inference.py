import time
import numpy as np
import onnxruntime
from pydub import AudioSegment
from transformers import AutoTokenizer
import torch
import torchaudio
import warnings

warnings.filterwarnings('ignore')


class FunASRNanoONNX:
    def __init__(self, model_path, sample_rate=16000, n_mels=80, nfft_stft=400,
                 window_length=400, hop_length=160, pre_emphasize=0.97,
                 use_normalizer=True, lfr_m=7, lfr_n=6, stop_tokens=None,
                 max_new_tokens=200, repeat_penalty=1.0, max_audio_len=512000,
                 num_threads=4):
        self.sample_rate = sample_rate
        self.use_normalizer = use_normalizer
        self.max_new_tokens = max_new_tokens
        self.repeat_penalty = repeat_penalty
        self.max_audio_len = max_audio_len
        self.stop_tokens = stop_tokens if stop_tokens else [151643, 151645]

        # ONNX Runtime会话
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        providers = ['CPUExecutionProvider']

        self.ort_encoder = onnxruntime.InferenceSession(
            f'{model_path}/encoder_adaptor.int8.onnx', opts, providers=providers)
        self.ort_embed = onnxruntime.InferenceSession(
            f'{model_path}/embedding.int8.onnx', opts, providers=providers)
        self.ort_decoder = onnxruntime.InferenceSession(
            f'{model_path}/decoder.int8.onnx', opts, providers=providers)

        # 解析decoder结构
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

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 特征提取器
        self.feat_extractor = AudioFeatureExtractor(
            n_mels, nfft_stft, window_length, hop_length,
            pre_emphasize, lfr_m, lfr_n, sample_rate)

        # 预计算系统嵌入
        sys_ids = self.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                                         return_tensors='np').astype(np.int64)
        user_ids = self.tokenizer.encode("<|im_start|>user\n", return_tensors='np').astype(np.int64)
        asst_ids = self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n",
                                          return_tensors='np').astype(np.int64)
        self.system_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: sys_ids})[0]
        self.user_prefix_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: user_ids})[0]
        self.assistant_prefix_embed = self.ort_embed.run([self.embed_out_name], {self.embed_in_name: asst_ids})[0]

    def _load_audio(self, path):
        audio = AudioSegment.from_file(path).set_frame_rate(self.sample_rate).set_channels(1)
        audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if self.use_normalizer:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio *= (8192.0 / rms)
            np.clip(audio, -32768.0, 32767.0, out=audio)
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]
        return audio

    def _encode_prompt(self, prompt):
        ids = self.tokenizer.encode(prompt, return_tensors='np').astype(np.int64)
        return self.ort_embed.run([self.embed_out_name], {self.embed_in_name: ids})[0]

    def _encode_audio(self, audio):
        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
        mel_feat = self.feat_extractor.extract(tensor).numpy().astype(np.float32)
        return self.ort_encoder.run(
            ['encoded_features'],
            {'speech': mel_feat, 'speech_lengths': np.array([mel_feat.shape[1]], dtype=np.int64)}
        )[0]

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

            # 贪心搜索
            penalized = logits * penalty
            next_token = np.argmax(penalized, axis=-1, keepdims=True).astype(np.int32)
            token_id = int(next_token[0, 0])
            if token_id in self.stop_tokens:
                break
            generated.append(token_id)
            penalty[0, token_id] *= self.repeat_penalty

            # 更新缓存
            for i in range(self.num_layers):
                inputs[f'in_key_{i}'] = outputs[i]
                inputs[f'in_value_{i}'] = outputs[i + self.num_layers]

            # 下一个token嵌入
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
        audio = self._load_audio(audio_path)
        audio_len = len(audio) / self.sample_rate

        audio_embed = self._encode_audio(audio)
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
        return text, len(token_ids), elapsed, audio_len


class AudioFeatureExtractor:
    def __init__(self, n_mels, nfft, win_len, hop_len, pre_emph, lfr_m, lfr_n, sr):
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.lfr_pad = (lfr_m - 1) // 2
        self.pre_emph = pre_emph
        self.hop_len = hop_len
        self.win_len = win_len
        self.nfft = nfft

        self.fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=nfft // 2 + 1, n_mels=n_mels, f_min=0.0, f_max=sr // 2,
            sample_rate=sr, norm='slaney', mel_scale='htk'
        ).transpose(0, 1).unsqueeze(0)

    def extract(self, audio_tensor):
        audio = audio_tensor.float().squeeze(1)  # (1, T)
        audio = audio - audio.mean()
        if self.pre_emph > 0:
            audio = torch.cat([audio[..., :1],
                               audio[..., 1:] - self.pre_emph * audio[..., :-1]], dim=-1)

        window = torch.hann_window(self.win_len)
        stft = torch.stft(audio, n_fft=self.nfft, hop_length=self.hop_len,
                          win_length=self.win_len, window=window,
                          center=True, onesided=True, return_complex=True)
        power = torch.abs(stft).pow(2)

        mel = torch.matmul(self.fbank, power)  # (1, 80, T)
        mel = (mel + 1e-7).log().transpose(1, 2)  # (1, T, 80)

        T = mel.shape[1]
        T_lfr = (T + self.lfr_n - 1) // self.lfr_n

        left_pad = mel[:, :self.lfr_pad, :].clone()
        if left_pad.shape[1] < self.lfr_pad:
            repeat = self.lfr_pad - left_pad.shape[1]
            left_pad = torch.cat([mel[:, :1, :].repeat(1, repeat, 1), left_pad], dim=1)
        padded = torch.cat([left_pad, mel], dim=1)

        lfr_list = []
        for t in range(T_lfr):
            start = t * self.lfr_n
            end = start + self.lfr_m
            if end > padded.shape[1]:
                start = max(0, padded.shape[1] - self.lfr_m)
                end = padded.shape[1]
            frames = padded[:, start:end, :]
            if frames.shape[1] < self.lfr_m:
                pad = self.lfr_m - frames.shape[1]
                frames = torch.cat([frames, frames[:, -1:, :].repeat(1, pad, 1)], dim=1)
            lfr_list.append(frames.reshape(1, -1))
        return torch.stack(lfr_list, dim=1)


if __name__ == "__main__":
    model_path = r'/path/to/Fun-ASR-Nano-2512-LLM-int8-onnx'

    test_audio = [
        model_path + "/example/zh_31s.wav",
        model_path + "/example/zh.mp3",
        model_path + "/example/en.mp3",
        model_path + "/example/ja.mp3",
        model_path + "/example/yue.mp3"
    ]
    hotwords = ['开放时间']
    hotwords_str = ", ".join(hotwords) if hotwords else ""
    languages = ["中文", "中文", "英文", "日语", "粤语"]
    if 'MLT' in model_path:
        test_audio.append(model_path + "/example/ko.mp3")
        languages.append("韩语")

    itn = True
    prompts = []
    for lang in languages:
        base = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n**上下文信息：**\n"
        if hotwords_str:
            base += f"热词列表：[{hotwords_str}]\n"
        base += f"语音转写成{lang}"
        if not itn:
            base += "，不进行文本规整"
        base += "："
        prompts.append(base)

    # max_audio_len=512000 # 32秒
    asr = FunASRNanoONNX(model_path, max_audio_len=512000, num_threads=8)

    for idx, (path, prompt) in enumerate(zip(test_audio, prompts)):
        print(f"\n--- 测试 {idx+1}: {path} ---")
        text, num_tokens, elapsed, audio_len = asr.transcribe(path, prompt)
        if text:
            print(f"音频时长: {audio_len:.2f}s")
            print(f"转写结果: {text}")
            print(f"生成 {num_tokens} tokens, 耗时 {elapsed:.2f}s, RTF={elapsed/audio_len:.3f}")
        else:
            print("未生成任何文本")