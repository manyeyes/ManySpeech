# FunASR-Nano ONNX模型推理使用文档

## 一、概述

FunASR-Nano 系列提供两种常用架构的 ONNX 模型，满足不同场景的语音识别需求：

- **CTC模型**：基于连接时序分类的端到端语音识别模型，推理速度快，适合实时性要求高的场景。
- **LLM模型**：基于大语言模型的语音识别模型，通过提示（prompt）引导输出，支持热词增强和上下文信息，适合需要灵活控制输出的场景。

本文档仅说明如何运行已放置在模型目录下的推理脚本，不包含代码内容。

---

## 二、环境准备

### 2.1 系统要求
- 操作系统：Linux、Windows 10+、macOS
- 硬件：CPU（建议4核以上）；可选NVIDIA GPU（需CUDA）
- 内存：≥4GB

### 2.2 依赖安装（Python 3.8–3.11）
```bash
pip install numpy onnxruntime==1.16.0 funasr transformers torch tiktoken
pip install soundfile librosa pydub  # 可选，增强音频格式支持
# GPU推理需安装 onnxruntime-gpu
```

---

## 三、模型下载与目录结构

两种模型均托管在 ModelScope，通过 `git clone` 下载。下载后目录结构如下（以 INT8 版本为例，FP32 版本文件名通常无 `.int8` 后缀）：

### CTC 模型目录结构
```
Fun-ASR-Nano-2512-CTC-int8-onnx/
├── encoder.int8.onnx          # 音频编码器
├── decoder.int8.onnx           # CTC解码器
├── multilingual.tiktoken        # 分词器文件
└── example/                     # 测试音频文件夹（可选）
    ├── zh.mp3
    └── ...
```

### LLM 模型目录结构
```
Fun-ASR-Nano-2512-LLM-int8-onnx/
├── encoder_adaptor.int8.onnx   # 音频编码器
├── embed.int8.onnx              # 文本嵌入层
├── decoder.int8.onnx             # 解码器
├── tokenizer.json                # HuggingFace tokenizer
├── vocab.json
├── tokens.txt
├── config.json
└── example/                      # 测试音频文件夹（可选）
    ├── zh_31s.wav
    └── ...
```

> **注意**：FP32 版本文件名不包含 `.int8`，如 `encoder.onnx`、`decoder.onnx`、`encoder_adaptor.onnx` 等。

---

## 四、CTC模型运行方法

### 4.1 放置推理脚本
请确保已将 CTC 模型推理脚本 `fun_asr_nano_ctc_onnx_inference.py` 放置于模型目录（如 `Fun-ASR-Nano-2512-CTC-int8-onnx/`）下。

### 4.2 修改配置
打开 `fun_asr_nano_ctc_onnx_inference.py`，找到配置区域，根据实际情况修改以下参数：

```python
CONFIG = {
    "model_dir": "分词器目录", # 例如 "/to/path/Fun-ASR-Nano-2512-CTC-onnx"
    "onnx_model_dir": "ONNX模型目录", # 例如 "/to/path/Fun-ASR-Nano-2512-CTC-onnx"
    "audio_test_path": "测试音频路径", # 例如 "/to/path/Fun-ASR-Nano-2512-CTC-onnx/example/zh.mp3"
    "blank_id_default": 60514,  # 与模型训练一致，不可乱改
    "intra_op_num_threads": 8,  # 按CPU核心数调整
    "inter_op_num_threads": 8,
    "audio_sample_rate": 16000  # 固定不可改
    "device_type": "CPU", # 指定设备
}
```

### 4.3 运行命令
在模型目录下执行：
```bash
cd Fun-ASR-Nano-2512-CTC-int8-onnx
python fun_asr_nano_ctc_onnx_inference.py
```

---

## 五、LLM模型运行方法

### 5.1 放置推理脚本
确保以下两个脚本已放置在 LLM 模型目录（如 `Fun-ASR-Nano-2512-LLM-int8-onnx/`）下：
- `fun_asr_nano_llm_onnx_inference.py` 

### 5.2 修改配置
打开 `fun_asr_nano_llm_onnx_inference.py`，修改以下配置：

```python
model_path = '/path/to/Fun-ASR-Nano-2512-LLM-int8-onnx'   # 当前目录

test_audio = [
    '/path/to/example/zh_31s.wav',                   # 测试音频
    '/path/to/example/zh.mp3',
    '/path/to/example/en.mp3',
    '/path/to/example/ja.mp3',
    '/path/to/example/yue.mp3'
]
prompt = "请将语音转写成中文："                    # 可自定义

# 初始化参数（如需GPU，修改类内部providers）
asr = FunASRNanoONNX(
    model_path=model_path,
    max_new_tokens=200,
    repeat_penalty=1.0,
    stop_tokens=[151643, 151645],
    max_audio_len=400000,
    num_threads=8,
)
```

若使用 FP32 版本，需在 `fun_asr_nano_llm_onnx_inference.py` 中修改 ONNX 文件名（去掉 `.int8` 后缀）。

### 5.3 运行命令
```bash
cd Fun-ASR-Nano-2512-LLM-int8-onnx
python fun_asr_nano_llm_onnx_inference.py
```

---

## 六、关键参数说明

| 模型 | 参数 | 说明 |
|------|------|------|
| CTC | `blank_id_default` | CTC空白标记ID（必须与训练一致，示例60514） |
| CTC | `encoder_file`/`decoder_file` | ONNX文件名（根据版本调整） |
| CTC | `tokenizer_file` | 分词器文件名（固定`multilingual.tiktoken`） |
| LLM | `max_new_tokens` | 最大生成token数 |
| LLM | `repeat_penalty` | 重复惩罚系数（>1.0抑制重复） |
| LLM | `stop_tokens` | 停止生成的token ID列表（默认`[151643,151645]`） |
| LLM | `max_audio_len` | 最大音频采样点数（16kHz下400000≈25秒） |
| 通用 | `intra_op_num_threads`/`num_threads` | ONNX Runtime线程数 |
| 通用 | `device_type`/`providers` | 推理设备（CPU/CUDA） |

---

## 七、注意事项

1. **模型文件版本**：INT8与FP32的文件名后缀不同，请根据实际下载版本修改脚本中的文件名。
2. **音频采样率**：必须为16000 Hz，脚本会自动重采样，但建议提前确认。
3. **长音频处理**：CTC模型输入通常不超过30秒，LLM模型建议单次音频≤25秒（可通过`max_audio_len`调整）。
4. **GPU推理**：需安装`onnxruntime-gpu`，并将ONNX会话的`providers`改为`['CUDAExecutionProvider']`。
5. **线程数**：建议根据CPU核心数设置，避免过载。

---

## 八、参考资源

- FunASR官方文档：[https://funasr.github.io/](https://funasr.github.io/)
- ModelScope模型主页：[https://www.modelscope.cn/models/manyeyes](https://www.modelscope.cn/models/manyeyes)
- ONNX Runtime文档：[https://onnxruntime.ai/](https://onnxruntime.ai/)

---
