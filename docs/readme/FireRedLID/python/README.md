# FireRedLID ONNX 推理脚本使用文档
## 一、简介
本工具基于 Python 实现 FireRedLID 模型的 ONNX 推理，支持音频语言识别（LID），无需依赖 `fireredlid` 自定义模块，集成了完整的特征提取、CMVN 归一化、束搜索解码逻辑，可直接运行。

## 二、环境依赖
### 安装命令
```bash
pip install kaldiio kaldi-native-fbank numpy torch onnxruntime
```
### 核心依赖说明
| 库名                | 作用                     |
|---------------------|--------------------------|
| kaldiio             | 读取音频文件、CMVN 数据  |
| kaldi-native-fbank  | 提取 Fbank 音频特征      |
| numpy/torch         | 数值计算、张量处理       |
| onnxruntime         | 运行 ONNX 模型推理       |

## 三、快速使用
### 1. 下载模型
```bash
# cd /path/to/workspace
git clone https://www.modelscope.cn/manyeyes/FireRedLID-int8-onnx.git
```

### 2. 配置参数
修改代码顶部的配置区域，替换为实际路径：
```python
# ======================== 配置参数 ========================
WAV_PATH = "/path/to/hello_zh.wav"       # 输入音频文件路径
MODEL_DIR = "/path/to/FireRedLID-int8-onnx"  # 模型目录（含cmvn.ark、dict.txt）
ENCODER_ONNX = os.path.join(MODEL_DIR, "encoder.int8.onnx")  # 编码器ONNX路径
DECODER_ONNX = os.path.join(MODEL_DIR, "decoder.int8.onnx")  # 解码器ONNX路径
USE_GPU = False                          # 是否使用GPU推理（需安装CUDA版onnxruntime）
BEAM_SIZE = 3                            # 束搜索宽度（建议3-5）
# =========================================================
```

### 3. 运行代码
```bash
python fireredlid_onnx_inference.py
```

### 4. 输出示例
```
========== 推理结果 ==========
音频文件: /path/to/hello_zh.wav
预测语言: zh
Token ID: 5
置信度: 0.9987
==================================
```

## 四、核心功能说明
### 1. 特征提取流程
1. 读取 WAV 音频文件，解析采样率和音频数据；
2. 提取 80 维 Fbank 特征（帧长25ms，帧移10ms）；
3. 基于 `cmvn.ark` 进行均值方差归一化；
4. 对特征进行 padding 对齐，生成批量特征张量。

### 2. 模型推理流程
1. 编码器（Encoder）：将音频特征编码为高维语义特征；
2. 解码器（Decoder）：采用两步束搜索解码，生成语言 Token；
3. 概率计算：通过 log softmax 计算候选 Token 概率，选择置信度最高的结果。

### 3. 关键文件说明
| 文件          | 作用                     |
|---------------|--------------------------|
| cmvn.ark      | 特征均值方差归一化参数   |
| dict.txt      | 语言 Token 与 ID 映射表  |
| encoder.int8.onnx | 量化后的编码器模型    |
| decoder.int8.onnx | 量化后的解码器模型    |

## 五、注意事项
1. 音频文件长度建议≥0.5秒，过短会导致特征提取失败；
2. 若需批量推理，可修改 `process_audio` 函数，支持多音频文件列表输入；
3. GPU 推理需满足：
   - 安装 `onnxruntime-gpu`（`pip install onnxruntime-gpu`）；
   - 系统已配置 CUDA 环境（CUDA 11.x 或 12.x）；
   - 将 `USE_GPU` 设置为 `True`。