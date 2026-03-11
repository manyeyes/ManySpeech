import os
import sys
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import onnx
import onnxruntime as ort

# ===================== 配置项（统一管理路径和参数） =====================
# 基础配置 - 可根据实际环境修改
CONFIG = {
    "tokenizer_dir": "/to/path/Fun-ASR-Nano-2512/onnx",
    "onnx_model_dir": "/to/path/Fun-ASR-Nano-2512/onnx",
    "audio_test_path": "/to/path/Fun-ASR-Nano-2512/example/zh.mp3", 
    "blank_id_default": 60514,
    "target_seq_len": 0,  # 大于0时限制解码时长(512≈30秒)
    "warmup_runs": 3,
    "benchmark_runs": 5,
    "intra_op_num_threads": 1,  # 根据CPU核心数调整
    "inter_op_num_threads": 1,  # 根据任务并行度调整
    "audio_sample_rate": 16000,  # 音频采样率
    "device_type": "CUDA" if torch.cuda.is_available() else "CPU",
}


# ===================== Tokenizer 注册 =====================
def SenseVoiceTokenizer(**kwargs):
    """SenseVoice分词器"""
    try:
        from funasr.models.sense_voice.whisper_lib.tokenizer import get_tokenizer
    except ImportError:
        raise ImportError("请安装 openai-whisper：pip install -U openai-whisper")

    language = kwargs.get("language", None)
    task = kwargs.get("task", None)
    is_multilingual = kwargs.get("is_multilingual", True)
    num_languages = kwargs.get("num_languages", 8749)
    vocab_path = kwargs.get("vocab_path", None)

    # 校验 vocab_path 是否存在
    if vocab_path and not os.path.exists(vocab_path):
        raise FileNotFoundError(f"指定的vocab文件不存在：{vocab_path}")

    tokenizer = get_tokenizer(
        multilingual=is_multilingual,
        num_languages=num_languages,
        language=language,
        task=task,
        vocab_path=vocab_path,
    )
    return tokenizer


# ===================== CTC推理器 =====================
class CTCInference:
    """CTC模型推理器（仅加载和推理ONNX模型）"""

    def __init__(
            self,
            encoder_onnx_path: str,
            decoder_onnx_path: str,
            blank_id: int = CONFIG["blank_id_default"],
            target_seq_len: int = CONFIG["target_seq_len"]
    ):
        """
        初始化推理器

        Args:
            encoder_onnx_path: 编码器ONNX路径
            decoder_onnx_path: 解码器ONNX路径
            blank_id: 空白标记ID
            target_seq_len: 目标序列长度
        """
        self.encoder_onnx_path = encoder_onnx_path
        self.decoder_onnx_path = decoder_onnx_path
        self.blank_id = blank_id
        self.target_seq_len = target_seq_len

        # 性能统计
        self.inference_stats = {
            "audio_load_time": 0,
            "feature_extract_time": 0,
            "encoder_infer_time": 0,
            "decoder_infer_time": 0,
            "decode_time": 0,
            "total_infer_time": 0
        }

        # 系统配置信息
        self.intra_op_num_threads = CONFIG["intra_op_num_threads"]
        self.inter_op_num_threads = CONFIG["inter_op_num_threads"]
        self.device_type = CONFIG["device_type"]
        self.audio_sample_rate = CONFIG["audio_sample_rate"]

        # 加载ONNX模型
        self.encoder_session = self._load_onnx_model(encoder_onnx_path, "编码器")
        self.decoder_session = self._load_onnx_model(decoder_onnx_path, "解码器")

        # 初始化音频前端处理
        self.frontend = self._init_frontend()

    def _load_onnx_model(self, model_path: str, model_type: str = "模型"):
        """加载ONNX模型"""
        if not os.path.exists(model_path):
            logging.error(f"{model_type}模型文件不存在: {model_path}")
            return None

        try:
            # 配置ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = self.intra_op_num_threads
            sess_options.inter_op_num_threads = self.inter_op_num_threads

            # 设置执行提供者
            providers = ['CUDAExecutionProvider'] if self.device_type == 'CUDA' else ['CPUExecutionProvider']

            # 加载模型
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )

            logging.info(f"{model_type} ONNX模型加载成功: {model_path}")
            return session

        except Exception as e:
            logging.error(f"{model_type} ONNX模型加载失败: {e}")
            return None

    def _init_frontend(self):
        """初始化音频前端处理"""
        try:
            from funasr.register import tables

            frontend_conf = {
                "fs": self.audio_sample_rate,
                "window": "hamming",
                "n_mels": 80,
                "frame_length": 25,
                "frame_shift": 10,
                "lfr_m": 7,
                "lfr_n": 6,
                "dither": 0,
                "snip_edges": True,
                "cmvn_file": None
            }

            frontend_class = tables.frontend_classes.get("wav_frontend")
            frontend = frontend_class(**frontend_conf)
            return frontend
        except Exception as e:
            logging.warning(f"初始化音频前端失败: {e}")
            return None

    def _pad_or_truncate_encoder_output(self, encoder_out_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """调整编码器输出长度到目标长度"""
        batch_size, seq_len, feat_dim = encoder_out_np.shape

        padded_encoder_out = np.zeros((batch_size, self.target_seq_len, feat_dim), dtype=np.float32)
        valid_len = min(seq_len, self.target_seq_len)
        padded_encoder_lens = np.array([valid_len] * batch_size, dtype=np.int64)
        padded_encoder_out[:, :valid_len, :] = encoder_out_np[:, :valid_len, :]

        logging.info(f"编码器输出长度调整: {seq_len} -> {valid_len} (目标长度: {self.target_seq_len})")

        return padded_encoder_out, padded_encoder_lens

    def calculate_rtf(self, infer_time: float, audio_duration: float) -> float:
        """计算实时因子（Real-Time Factor）"""
        if audio_duration <= 0:
            return float('inf')

        rtf = infer_time / audio_duration
        logging.info(f"RTF计算: 推理耗时={infer_time:.3f}s, 音频时长={audio_duration:.3f}s, RTF={rtf:.4f}")
        return rtf

    def _load_and_process_audio(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """加载并处理音频文件，提取特征"""
        try:
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

            # 加载音频
            load_start = time.time()
            data_src = load_audio_text_image_video(audio_path, fs=self.audio_sample_rate)
            audio_duration = len(data_src) / self.audio_sample_rate
            load_time = time.time() - load_start

            # 提取声学特征
            feat_start = time.time()
            speech, speech_lengths = extract_fbank(
                data_src,
                data_type="sound",
                frontend=self.frontend,
                is_final=True,
            )
            feat_time = time.time() - feat_start

            # 转换为numpy数组
            speech_np = speech.cpu().numpy().astype(np.float32)
            speech_lengths_np = speech_lengths.cpu().numpy().astype(np.int64)

            # 更新统计信息
            self.inference_stats["audio_load_time"] = load_time
            self.inference_stats["feature_extract_time"] = feat_time

            logging.info(
                f"音频处理完成: 时长={audio_duration:.2f}s, 采样率={self.audio_sample_rate}Hz, 特征形状={speech_np.shape}")

            return speech_np, speech_lengths_np, audio_duration

        except Exception as e:
            logging.error(f"音频处理失败: {e}")
            raise

    def _decode_ctc_logits(
            self,
            ctc_logits: np.ndarray,
            lengths: np.ndarray,
            tokenizer=None
    ) -> List[Dict]:
        """解码CTC logits为文本"""
        results = []
        batch_size = ctc_logits.shape[0]

        for i in range(batch_size):
            seq_len = min(lengths[i], ctc_logits.shape[1])
            logits = ctc_logits[i, :seq_len, :]

            # 贪心解码
            yseq = np.argmax(logits, axis=-1)

            # 移除重复和空白标记
            prev_token = -1
            decoded_tokens = []
            for token in yseq:
                if token != prev_token and token != self.blank_id:
                    decoded_tokens.append(token)
                prev_token = token

            # 解码为文本
            text = ""
            if tokenizer is not None:
                try:
                    text = tokenizer.decode(decoded_tokens)
                except Exception as e:
                    logging.warning(f"tokenizer解码失败: {e}")
                    text = f"Tokens: {decoded_tokens[:20]}..."  # 截断过长的token列表
            else:
                text = f"Tokens: {decoded_tokens[:20]}..."

            results.append({
                "text": text,
                "tokens": decoded_tokens,
                "raw_tokens": yseq.tolist(),
                "sequence_length": seq_len
            })

        return results

    def inference_from_audio(
            self,
            audio_path: str,
            tokenizer=None,
            warmup_runs: int = CONFIG["warmup_runs"],
            benchmark_runs: int = CONFIG["benchmark_runs"]
    ) -> Dict:
        """
        从音频文件进行推理

        Args:
            audio_path: 音频文件路径
            tokenizer: 分词器（可选）
            warmup_runs: 预热次数
            benchmark_runs: 基准测试次数

        Returns:
            推理结果字典（包含系统配置信息）
        """
        # 检查模型是否加载成功
        if self.encoder_session is None or self.decoder_session is None:
            return {"error": "编码器或解码器未加载"}

        try:
            # 1. 加载并处理音频
            speech_np, speech_lengths_np, audio_duration = self._load_and_process_audio(audio_path)

            # 2. 预热运行（避免首次推理耗时偏高）
            if warmup_runs > 0:
                logging.info(f"开始预热运行（{warmup_runs}次）...")
                for i in range(warmup_runs):
                    try:
                        # 编码器推理
                        encoder_inputs = {"speech": speech_np, "speech_lengths": speech_lengths_np}
                        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
                        encoder_out_np = encoder_outputs[0]
                        encoder_out_lens_np=encoder_outputs[1]

                        # 调整长度
                        if self.target_seq_len>0:
                            encoder_out_np, encoder_out_lens_np = self._pad_or_truncate_encoder_output(encoder_out_np)

                        # 解码器推理
                        decoder_inputs = {"encoder_out": encoder_out_np, "encoder_out_lens": encoder_out_lens_np}
                        self.decoder_session.run(None, decoder_inputs)
                    except Exception as e:
                        logging.warning(f"预热运行出错: {e}")
                logging.info("预热完成")

            # 3. 基准测试
            infer_times = []
            encoder_times = []
            decoder_times = []
            ctc_logits = None
            output_lengths = None

            logging.info(f"开始基准测试（{benchmark_runs}次）...")
            for i in range(benchmark_runs):
                start_time = time.time()

                # 编码器推理
                enc_start = time.time()
                encoder_inputs = {"speech": speech_np, "speech_lengths": speech_lengths_np}
                encoder_outputs = self.encoder_session.run(None, encoder_inputs)
                encoder_out_np = encoder_outputs[0]
                encoder_out_lens_np = encoder_outputs[1]
                enc_time = time.time() - enc_start
                encoder_times.append(enc_time)

                # 调整编码器输出长度
                if self.target_seq_len>0:
                    encoder_out_np, encoder_out_lens_np = self._pad_or_truncate_encoder_output(encoder_out_np)

                # 解码器推理
                dec_start = time.time()
                decoder_inputs = {"encoder_out": encoder_out_np, "encoder_out_lens": encoder_out_lens_np}
                decoder_outputs = self.decoder_session.run(None, decoder_inputs)
                ctc_logits = decoder_outputs[0]
                output_lengths = decoder_outputs[1]
                dec_time = time.time() - dec_start
                decoder_times.append(dec_time)

                # 总推理时间
                total_time = time.time() - start_time
                infer_times.append(total_time)

            # 计算平均耗时
            avg_infer_time = np.mean(infer_times)
            avg_encoder_time = np.mean(encoder_times)
            avg_decoder_time = np.mean(decoder_times)

            # 更新统计信息
            self.inference_stats.update({
                "encoder_infer_time": avg_encoder_time,
                "decoder_infer_time": avg_decoder_time,
                "total_infer_time": avg_infer_time
            })

            logging.info(f"基准测试完成:")
            logging.info(f"  平均推理耗时: {avg_infer_time:.3f}s (±{np.std(infer_times):.3f})")
            logging.info(f"  平均编码器耗时: {avg_encoder_time:.3f}s")
            logging.info(f"  平均解码器耗时: {avg_decoder_time:.3f}s")

            # 4. 解码为文本
            decode_start = time.time()
            results = self._decode_ctc_logits(ctc_logits, output_lengths, tokenizer)
            self.inference_stats["decode_time"] = time.time() - decode_start

            # 5. 计算RTF
            rtf = self.calculate_rtf(avg_infer_time, audio_duration)

            # 6. 整理并返回结果（包含系统配置信息）
            return {
                # 基础信息
                "audio_path": audio_path,
                "audio_duration": audio_duration,
                "audio_sample_rate": self.audio_sample_rate,
                "speech_features_shape": speech_np.shape,
                "encoder_output_shape": encoder_out_np.shape,
                "ctc_logits_shape": ctc_logits.shape if ctc_logits is not None else None,

                # 推理结果
                "predictions": results,

                # 性能指标
                "inference_time": avg_infer_time,
                "encoder_time": avg_encoder_time,
                "decoder_time": avg_decoder_time,
                "feature_extract_time": self.inference_stats["feature_extract_time"],
                "audio_load_time": self.inference_stats["audio_load_time"],
                "decode_time": self.inference_stats["decode_time"],
                "rtf": rtf,
                "batch_size": speech_np.shape[0],
                "benchmark_runs": benchmark_runs,
                "infer_time_std": np.std(infer_times),

                # 系统配置
                "device_type": self.device_type,
                "intra_op_num_threads": self.intra_op_num_threads,
                "inter_op_num_threads": self.inter_op_num_threads
            }

        except Exception as e:
            logging.error(f"音频推理失败: {e}")
            logging.error(traceback.format_exc())
            return {"error": str(e), "audio_path": audio_path}


# ===================== 工具函数 =====================
def get_tokenizer(tokenizer_dir: str) -> Optional[object]:
    """获取tokenizer（用于推理）"""
    try:
        vocab_path = os.path.join(tokenizer_dir, "multilingual.tiktoken")
        tokenizer = SenseVoiceTokenizer(
            language="en",
            task="transcribe",
            vocab_path=vocab_path
        )
        return tokenizer
    except Exception as e:
        logging.warning(f"获取tokenizer失败: {e}")
        return None


# ===================== 主函数 =====================
def main():
    """主函数：加载ONNX模型并进行推理测试"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1. 检查ONNX模型路径
    encoder_onnx_path = os.path.join(CONFIG["onnx_model_dir"], "encoder.onnx")
    decoder_onnx_path = os.path.join(CONFIG["onnx_model_dir"], "decoder.onnx")
    encoder_int8_path = os.path.join(CONFIG["onnx_model_dir"], "encoder.int8.onnx")
    decoder_int8_path = os.path.join(CONFIG["onnx_model_dir"], "decoder.int8.onnx")

    # 检查必要的模型文件
    if not os.path.exists(encoder_onnx_path):
        logging.error(f"编码器ONNX模型不存在: {encoder_onnx_path}")
        # return

    if not os.path.exists(decoder_onnx_path):
        logging.error(f"解码器ONNX模型不存在: {decoder_onnx_path}")
        # return

    # 2. 获取tokenizer
    tokenizer = get_tokenizer(CONFIG["tokenizer_dir"])

    # 3. 检查测试音频
    if not os.path.exists(CONFIG["audio_test_path"]):
        logging.error(f"测试音频文件不存在: {CONFIG['audio_test_path']}")
        print(f"\n❌ 错误：音频文件不存在，请检查路径: {CONFIG['audio_test_path']}")
        return

    # 4. FP32 ONNX模型推理测试
    print("\n" + "=" * 50)
    print("FP32 ONNX模型推理测试")
    print("=" * 50)

    inference_original = CTCInference(
        encoder_onnx_path=encoder_onnx_path,
        decoder_onnx_path=decoder_onnx_path,
        blank_id=CONFIG["blank_id_default"],
        target_seq_len=CONFIG["target_seq_len"]
    )

    results_original = inference_original.inference_from_audio(
        audio_path=CONFIG["audio_test_path"],
        tokenizer=tokenizer,
        warmup_runs=CONFIG["warmup_runs"],
        benchmark_runs=CONFIG["benchmark_runs"]
    )

    # 打印FP32 ONNX模型结果（包含新增的系统信息）
    if "error" not in results_original:
        print(f"\n📊 FP32 ONNX模型:")
        print(f"  🖥️  设备类型: {results_original['device_type']}")
        print(
            f"  🧵  线程配置: intra_op={results_original['intra_op_num_threads']}, inter_op={results_original['inter_op_num_threads']}")
        print(
            f"  🎵  音频信息: 时长={results_original['audio_duration']:.2f}s, 采样率={results_original['audio_sample_rate']}Hz")
        print(f"  ⏱️  推理耗时: {results_original['inference_time']:.3f}s (±{results_original['infer_time_std']:.3f})")
        print(f"  🚀  RTF: {results_original['rtf']:.4f}")
        print(f"  📝  解码结果: {results_original['predictions'][0]['text']}")
    else:
        print(f"\n❌ FP32 ONNX模型推理失败: {results_original['error']}")

    # 5. 量化模型推理测试（如果存在）
    if os.path.exists(encoder_int8_path) and os.path.exists(decoder_int8_path):
        print("\n" + "=" * 50)
        print("INT8量化ONNX模型推理测试（编码器+解码器）")
        print("=" * 50)

        inference_int8 = CTCInference(
            encoder_onnx_path=encoder_int8_path,  # 使用量化的编码器
            decoder_onnx_path=decoder_int8_path,
            blank_id=CONFIG["blank_id_default"],
            target_seq_len=CONFIG["target_seq_len"]
        )

        results_int8 = inference_int8.inference_from_audio(
            audio_path=CONFIG["audio_test_path"],
            tokenizer=tokenizer,
            warmup_runs=CONFIG["warmup_runs"],
            benchmark_runs=CONFIG["benchmark_runs"]
        )

        # 打印量化模型结果（包含新增的系统信息）
        if "error" not in results_int8:
            print(f"\n📊 INT8量化模型:")
            print(f"  🖥️  设备类型: {results_int8['device_type']}")
            print(
                f"  🧵  线程配置: intra_op={results_int8['intra_op_num_threads']}, inter_op={results_int8['inter_op_num_threads']}")
            print(
                f"  🎵  音频信息: 时长={results_int8['audio_duration']:.2f}s, 采样率={results_int8['audio_sample_rate']}Hz")
            print(f"  ⏱️  推理耗时: {results_int8['inference_time']:.3f}s (±{results_int8['infer_time_std']:.3f})")
            print(f"  🚀  RTF: {results_int8['rtf']:.4f}")
            print(f"  📝  解码结果: {results_int8['predictions'][0]['text']}")

            # 性能对比
            if "error" not in results_original:
                speedup = (results_original['inference_time'] - results_int8['inference_time']) / results_original[
                    'inference_time'] * 100
                rtf_improvement = (results_original['rtf'] - results_int8['rtf']) / results_original['rtf'] * 100

                print("\n" + "=" * 50)
                print("性能对比结果")
                print("=" * 50)
                print(f"\n🚀 性能对比:")
                print(f"  🚄  推理速度提升: {speedup:.1f}%")
                print(f"  📉  RTF改善: {rtf_improvement:.1f}%")
        else:
            print(f"\n❌ 量化模型推理失败: {results_int8['error']}")
    elif os.path.exists(decoder_int8_path):
        # 兼容仅解码器量化的情况
        print("\n" + "=" * 50)
        print("INT8量化ONNX模型推理测试（仅解码器）")
        print("=" * 50)

        inference_int8 = CTCInference(
            encoder_onnx_path=encoder_onnx_path,
            decoder_onnx_path=decoder_int8_path,
            blank_id=CONFIG["blank_id_default"],
            target_seq_len=CONFIG["target_seq_len"]
        )

        results_int8 = inference_int8.inference_from_audio(
            audio_path=CONFIG["audio_test_path"],
            tokenizer=tokenizer,
            warmup_runs=CONFIG["warmup_runs"],
            benchmark_runs=CONFIG["benchmark_runs"]
        )

        # 打印量化模型结果
        if "error" not in results_int8:
            print(f"\n📊 INT8量化模型（仅解码器）:")
            print(f"  🖥️  设备类型: {results_int8['device_type']}")
            print(
                f"  🧵  线程配置: intra_op={results_int8['intra_op_num_threads']}, inter_op={results_int8['inter_op_num_threads']}")
            print(
                f"  🎵  音频信息: 时长={results_int8['audio_duration']:.2f}s, 采样率={results_int8['audio_sample_rate']}Hz")
            print(f"  ⏱️  推理耗时: {results_int8['inference_time']:.3f}s (±{results_int8['infer_time_std']:.3f})")
            print(f"  🚀  RTF: {results_int8['rtf']:.4f}")
            print(f"  📝  解码结果: {results_int8['predictions'][0]['text']}")

            # 性能对比
            if "error" not in results_original:
                speedup = (results_original['inference_time'] - results_int8['inference_time']) / results_original[
                    'inference_time'] * 100
                rtf_improvement = (results_original['rtf'] - results_int8['rtf']) / results_original['rtf'] * 100

                print("\n" + "=" * 50)
                print("性能对比结果")
                print("=" * 50)
                print(f"\n🚀 性能对比:")
                print(f"  🚄  推理速度提升: {speedup:.1f}%")
                print(f"  📉  RTF改善: {rtf_improvement:.1f}%")
        else:
            print(f"\n❌ 量化模型推理失败: {results_int8['error']}")
    else:
        logging.info("INT8量化模型不存在，跳过量化模型测试")
        if not os.path.exists(encoder_int8_path):
            logging.info(f"编码器INT8模型不存在: {encoder_int8_path}")
        if not os.path.exists(decoder_int8_path):
            logging.info(f"解码器INT8模型不存在: {decoder_int8_path}")


if __name__ == "__main__":
    main()