# FireRedASR2-AED ONNX 推理脚本使用文档

## 一、简介

本工具基于 Python 实现 **FireRedASR2-AED** 模型的 ONNX 推理，支持端到端语音识别（中文/英文），集成了完整的特征提取（Fbank + CMVN）、ONNX 模型加载、贪心解码以及 CTC 强制对齐生成词级时间戳的功能。无需依赖原始 PyTorch 训练代码，可直接运行。

## 二、环境依赖

### 安装命令
```bash
pip install kaldiio kaldi-native-fbank numpy torch torchaudio onnxruntime
```

### 核心依赖说明
| 库名                | 作用                            |
|---------------------|---------------------------------|
| kaldiio             | 读取音频文件、CMVN 统计数据     |
| kaldi-native-fbank  | 提取 80 维 Fbank 音频特征       |
| numpy / torch       | 数值计算、张量处理              |
| torchaudio          | 提供 CTC 强制对齐功能            |
| onnxruntime         | 运行 ONNX 模型推理（Encoder/Decoder/CTC） |

## 三、快速使用

### 1. 下载模型
从 ModelScope 下载 ONNX 格式的模型文件（int8 量化版）：
```bash
git clone https://www.modelscope.cn/manyeyes/fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212.git
```
或者（fp32 版）：
```bash
git clone https://www.modelscope.cn/manyeyes/fireredasr2-aed-large-zh-en-onnx-offline-20260212.git
```
将下载的文件夹放置在工作目录，例如 `/path/to/workspace/fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212`。

### 2. 配置参数
修改脚本顶部的配置区域，替换为实际路径：
```python
# ======================== 配置参数 ========================
WAV_PATH = "/path/to/your/audio.wav"                      # 待识别音频文件
ONNX_DIR = "/path/to/fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212"  # 模型根目录
USE_GPU = False                                            # 是否使用 GPU（需安装 onnxruntime-gpu）
MAX_LEN_RATIO = 1.0                                        # 解码最大长度相对于编码帧数的比例
# ==========================================================
```
其他模型内部参数（`d_model`、`vocab_size` 等）已在脚本中预设，与模型匹配，一般无需修改。

### 3. 运行代码
```bash
# without ctc(fireredasr-aed)
python fireredasr_onnx_inference.py
# with ctc(fireredasr2-aed)
python fireredasr2_onnx_inference.py
```

### 4. 输出示例
```
识别文本: 甚至出现交易几乎停滞的情况
时间戳 (秒):
  甚: 0.520 - 0.640
  至: 0.720 - 0.840
  出: 1.000 - 1.120
  现: 1.200 - 1.320
  交: 1.520 - 1.680
  易: 1.760 - 1.840
  几: 2.040 - 2.120
  乎: 2.240 - 2.320
  停: 2.480 - 2.600
  滞: 2.720 - 2.800
  的: 2.920 - 3.040
  情: 3.120 - 3.280
  况: 3.360 - 3.480
```

## 四、核心功能说明

### 1. 特征提取流程
1. 使用 `kaldiio` 读取 WAV 音频文件，获取采样率和音频数据；
2. 通过 `kaldi-native-fbank` 提取 80 维 Fbank 特征（帧长 25ms，帧移 10ms）；
3. 加载 `cmvn.ark` 文件对特征进行均值方差归一化；
4. 对同一批次的特征进行 padding 对齐，生成 `(batch, time, 80)` 的张量。

### 2. 模型推理流程
1. **Encoder**：将音频特征输入编码器 ONNX 模型，输出高层语义特征 `enc_out`（时间维度下采样 4 倍，即帧移 40ms）及对应的掩码；
2. **Decoder**：采用自回归贪心解码，初始输入为 `<sos>` 标记，每一步使用当前已生成的 token 序列和缓存状态（cache）调用解码器 ONNX 模型，得到下一个 token 的概率分布，取 argmax 作为当前步输出，直到遇到 `<eos>` 或达到最大长度；
3. **CTC**：将编码器输出送入 CTC ONNX 模型，获得帧级 logits；
4. **强制对齐**：利用 `torchaudio.functional.forced_align` 将解码出的 token 序列与 CTC logits 对齐，生成每个 token 的起止时间（精度为 40ms）。

### 3. 关键文件说明
| 文件                 | 作用                                    |
|----------------------|-----------------------------------------|
| `cmvn.ark`           | 特征均值方差归一化参数                  |
| `tokens.txt`         | 字符（或子词）与 ID 的映射表            |
| `encoder.int8.onnx`  | 量化后的编码器模型                      |
| `decoder.int8.onnx`  | 量化后的解码器模型（含缓存输入）        |
| `ctc.int8.onnx`      | 量化后的 CTC 输出层模型                 |

## 五、注意事项
1. **音频格式**：支持 WAV 等 `kaldiio` 可读格式，采样率自动检测（推荐 16kHz）；
2. **音频时长**：建议 ≥ 0.5 秒，过短可能导致特征不足；
3. **解码长度**：默认最大输出长度等于编码帧数（`MAX_LEN_RATIO=1.0`），可根据实际需要调整（例如设为 1.2 以生成更长文本）；
4. **GPU 推理**：
   - 安装 `onnxruntime-gpu`（`pip install onnxruntime-gpu`）；
   - 系统需配置 CUDA（11.x 或 12.x）；
   - 将 `USE_GPU = True`；
5. **时间戳精度**：受 CTC 帧移（40ms）限制，边界可能有 ±40ms 误差。