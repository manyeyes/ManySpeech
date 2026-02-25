# 音频分类ONNX模型推理脚本使用说明
## 简介
基于 CED 系列（[ced-tiny](https://huggingface.co/mispeech/ced-mini/tree/main)/[ced-mini](https://huggingface.co/mispeech/ced-mini/tree/main)/[ced-small](https://huggingface.co/mispeech/ced-mini/tree/main)/[ced-base](https://huggingface.co/mispeech/ced-mini/tree/main)）ONNX 音频分类模型，实现对输入音频的类别预测，自动输出音频所属类别及对应概率（Top-K 结果），支持 GPU/CPU 推理切换，适配自定义标签映射文件。


## 安装依赖
```bash
# 基础依赖（CPU推理）
pip install torch torchaudio onnxruntime numpy pathlib
# GPU加速（替换onnxruntime，需匹配CUDA版本）
pip install torch torchaudio onnxruntime-gpu numpy pathlib
```

## 准备文件
需要以下三类文件：
- **ONNX 模型文件**：如 `model.onnx`
- **标签映射文件**：`tokens.txt`（每行一个类别标签，与模型输出ID对应）
- **待推理音频文件**：16kHz 单声道 WAV 格式（非该格式脚本会自动转换采样率）

## 使用
### 1. 准备onnx模型文件
```bash
git clone https://www.modelscope.cn/manyeyes/ced-tiny-audio-tagging-onnx.git

git clone https://www.modelscope.cn/manyeyes/ced-mini-audio-tagging-onnx.git

git clone https://www.modelscope.cn/manyeyes/ced-small-audio-tagging-onnx.git

git clone https://www.modelscope.cn/manyeyes/ced-base-audio-tagging-onnx.git
```

### 2. 修改配置
在脚本末尾的 `__main__` 部分，替换为实际文件路径：
```python
# 需修改的路径配置
LABEL_TOKENS = "/path/to/tokens.txt"    # 标签映射文件路径
ONNX_MODEL_PATH = "/path/to/model.onnx" # ONNX模型文件路径
AUDIO_PATH = "/path/to/1.wav"          # 待推理音频文件路径
TOP_K = 3                              # 输出前K个预测结果
```

### 3. 运行测试
```bash
python ced_onnx_inference.py
```
输出示例：
```
Top-k predictions (ONNX):
Top1: Cat                            0.9342
Top2: Animal                         0.9114
Top3: Domestic animals, pets         0.8899
```

### 4. 代码调用
```python
from ced_onnx_inference import TokenConverter, onnx_inference

# 1. 加载标签映射
token_converter = TokenConverter("/path/to/tokens.txt")

# 2. 执行推理（可直接调用核心函数）
results = onnx_inference(
    onnx_model_path="/path/to/model.onnx",
    audio_path="/path/to/1.wav",
    token_converter=token_converter,
    top_k=3
)

# 3. 解析结果
print("推理结果：")
for idx, (label, prob) in enumerate(results):
    print(f"第{idx+1}名：{label}（概率：{prob:.4f}）")
```

## 核心参数说明
| 参数名          | 取值    | 说明                                  |
|-----------------|---------|---------------------------------------|
| SR              | 16000   | 模型要求的音频采样率（固定值）        |
| N_MELS          | 64      | Mel频谱维度（与模型训练一致）         |
| TARGET_LENGTH   | 1012    | 模型期望的时间帧数（对应10秒音频）    |
| TOP_K           | 3       | 输出前K个最高概率的类别（可自定义）   |
| providers       | -       | 推理设备，`CUDAExecutionProvider`（GPU）/`CPUExecutionProvider`（CPU） |

## 注意事项
### 1. 音频格式要求
- 推荐输入 16kHz 单声道 WAV 音频，非该采样率脚本会自动转换，但可能影响推理精度；
- 支持 torchaudio 可读取的所有格式（如 flac、mp3 等，需确保安装对应解码器）。

### 2. 标签文件
- `tokens.txt` 每行一个标签，无空行，行号即id；
- `tokens_zh.txt` 翻译后的中文标签；。

### 3. 模型优化建议
- 若需提升推理速度，可先使用 `onnxsim` 简化模型：
  ```bash
  pip install onnxsim
  python -m onnxsim /path/to/model.onnx /path/to/model_optimized.onnx
  ```
  简化后将 `ONNX_MODEL_PATH` 改为简化后的模型路径即可。

### 4. 批量推理扩展
如需批量处理音频，可循环调用 `onnx_inference` 函数：
```python
import os

# 批量音频目录
audio_dir = "/path/to/audio_dir"
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]

# 循环推理
for audio_file in audio_files:
    print(f"\n===== 处理音频：{audio_file} =====")
    onnx_inference(ONNX_MODEL_PATH, audio_file, token_converter, TOP_K)
```