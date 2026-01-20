# FunASR-Nano ONNX模型推理使用文档

# 一、概述

本文档为FunASR-Nano ONNX模型python版本的推理使用指南，仅覆盖必备操作，适用于测试和快速部署。

# 二、环境准备

## 2.1 系统要求及推荐配置

- 系统：Linux（Ubuntu 18.04+/CentOS 7+）、Windows 10+

- 硬件：CPU（x86_64，4核+）；可选NVIDIA GPU（配套CUDA/CuDNN）

- 内存：≥4GB可用内存

## 2.2 依赖安装（Python 3.8-3.11）

```bash

# 核心依赖
pip install numpy torch==1.18.0+cpu onnx onnxruntime==1.16.0 funasr
# 可选
pip openai-whisper
# 音频依赖（加载失败时补充）
pip install soundfile librosa
```

注：GPU场景需安装对应CUDA版本PyTorch及onnxruntime-gpu。

# 三、使用步骤

## 3.1 准备文件

1. ONNX模型：encoder.onnx（必选）、decoder.onnx（必选）；INT8量化版（可选，需成对/单解码器使用）

2. 分词器：multilingual.tiktoken（放模型目录）

3. 音频：MP3/WAV格式，采样率固定16000Hz

## 3.2 配置参数（CONFIG字典）

仅需调整核心路径，关键参数保持默认即可：

```python
# 配置说明
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

## 3.3 运行推理

```bash
cd /to/path

# 下载 fp32 模型
git clone https://www.modelscope.cn/manyeyes/Fun-ASR-Nano-2512-CTC-onnx.git

# 或者下载 int 模型
git clone https://www.modelscope.cn/manyeyes/Fun-ASR-Nano-2512-CTC-int8-onnx.git

# 以 fp32 模型为例
cd Fun-ASR-Nano-2512-CTC-onnx

# 将 inferencer.py 
# 拷贝到 Fun-ASR-Nano-2512-CTC-onnx 文件夹

# 运行推理
python inferencer.py 
```

自动完成模型检查、推理、结果输出，含原始/量化模型（若存在）性能对比。

# 四、关键参数说明

|参数名|核心说明|禁忌|
|---|---|---|
|blank_id_default|CTC解码空白标记ID|必须与模型训练一致，否则解码异常|
|audio_sample_rate|音频采样率|固定16000Hz，修改会导致识别错误|
|线程参数|控制CPU并行效率|避免超过CPU核心数，防止性能下降|
# 五、注意事项

1. 模型需完整配对，版本与ONNX Runtime兼容（推荐1.14.0+）。

2. GPU推理需安装onnxruntime-gpu，确保CUDA/CuDNN版本匹配。

3. 音频需保证格式正常、采样率正确，长音频建议分段处理（分段最大长度30秒）。

4. 量化模型提速15%左右，可能有少量精度损失，按需选择。

5. 推理过慢：调优线程数、使用量化模型或GPU。

