# FireRedVad ONNX 语音活动检测脚本使用说明

## 简介
基于 FireRedVad ONNX 模型，对输入音频进行语音活动检测（VAD），自动检测出音频中的语音段起止时间（秒），支持流式和非流式处理。

## 安装依赖
```bash
pip install numpy onnxruntime soundfile kaldi-native-fbank kaldiio
```
如需 GPU 加速，用 `onnxruntime-gpu` 替换 `onnxruntime`。

## 准备模型文件
需要以下两个文件：
- **ONNX 模型**：如 `model.onnx`
- **CMVN 文件**：`cmvn.ark`（Kaldi 格式，用于特征归一化）

## 使用
1. **下载模型**：
    ```bash
    cd /to/path

    # 示例：从 Modelscope 下载
    git clone https://modelscope.cn/models/manyeyes/FireRedVad-onnx.git

    # 进入模型目录
    cd FireRedVad-onnx
    ```
2. **准备推理脚本**：将以下完整脚本保存为 `fireredvad_onnx_inference.py`（或直接下载示例脚本）。

3. **修改脚本配置**：在脚本末尾的 `__main__` 部分，将以下路径改为实际路径：
   ```python
    onnx_path = "/path/to/FireRedVad-onnx/model.onnx"
    cmvn_path = "/path/to/FireRedVad-onnx/cmvn.ark"
    audio_file = "/path/to/FireRedVad-onnx/0.wav"
   ```
4. **运行测试**：
   ```bash
   python fireredvad_onnx_inference.py
   ```
   输出示例：
   ```
   特征形状: (1235, 80), 音频时长: 12.369s
   概率统计: min=0.0005, max=0.3112, mean=0.0423
   超过阈值的帧数: 30 / 1235
   音频时长: 12.369 秒
   检测到的语音段:
     0.070 - 5.350
     6.250 - 9.540
     10.050 - 12.100
   ```

5. **代码调用**：
   ```python
   from fireredvad_onnx_inference import ONNXFireRedVad, FireRedVadConfig

   # 配置参数（可根据需要调整）
   config = FireRedVadConfig(
       use_gpu=False,
       smooth_window_size=5,
       speech_threshold=0.4,
       min_speech_frame=20,
       max_speech_frame=2000,
       min_silence_frame=20,
       merge_silence_frame=20,
       extend_speech_frame=20
   )

   # 初始化 VAD 引擎
   vad = ONNXFireRedVad(
       onnx_path="model.onnx",
       config=config,
       cmvn_path="cmvn.ark"
   )

   # 对音频文件进行检测
   result, probs = vad.detect("test.wav")
   print(f"音频时长: {result['dur']} 秒")
   for seg in result['timestamps']:
       print(f"  {seg[0]:.3f} - {seg[1]:.3f}")
   ```

## 注意事项
- **音频格式**：输入音频需为 16kHz 单声道，支持常见格式（wav、flac 等），脚本会自动读取。
- **特征提取**：内部使用 80 维 FBank（帧长 25ms，帧移 10ms），与训练时一致。
- **后处理参数**：
  - `speech_threshold`：语音概率阈值，默认 0.4，可根据实际输出概率调整（如 0.3~0.4）。
  - `min_speech_frame`：最短语音帧数（20 帧 = 0.2 秒），低于此值的语音段被丢弃。
  - `min_silence_frame`：最短静音帧数，用于合并短停顿。
  - `extend_speech_frame`：语音段前后扩展帧数，可补偿边界偏移。
- **GPU 推理**：如需 GPU 加速，在配置中设置 `use_gpu=True`，并确保已安装 `onnxruntime-gpu`。