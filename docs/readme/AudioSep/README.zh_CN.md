# AudioSep

一款基于C#开发的高性能音频处理库，专注于音频分离、降噪与增强功能。它支持多种运行环境，包括**.NET Framework 4.6.1+**、**.NET 6.0+**、**.NET Core 3.1** 及 **.NET Standard 2.0+**，可实现跨平台编译与AOT（提前编译，Ahead-of-Time Compilation），且使用简单便捷。


## 支持的模型（ONNX）

| 模型名称 | 类型 | 参考 | 功能 | 下载地址  |
| ---------- | ---------- | ------------ | ---------- | ---------- |
| ZipEnhancer-se-16k-base-onnx | 非流式     | clearervoice | 语音降噪   | [modelscope](https://modelscope.cn/models/manyeyes/ZipEnhancer-se-16k-base-onnx "modelscope") |
| spleeter-2stems-44k-onnx   | 非流式     | spleeter     | 语音分离   | [modelscope](https://modelscope.cn/models/manyeyes/spleeter-2stems-44k-onnx "modelscope")     |
| gtcrn-se-16k-onnx-offline  | 非流式     | gtcrn        | 语音降噪   | [modelscope](https://modelscope.cn/models/manyeyes/gtcrn-se-16k-onnx-offline "modelscope")    |
| mossformer2-se-48k-onnx    | 非流式     | clearervoice | 语音降噪   | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-se-48k-onnx "modelscope")      |
| mossformer2-sr-48k-onnx    | 非流式     | clearervoice | 语音增强   | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-sr-48k-onnx "modelscope")      |
| mossformer2-ss-16k-onnx    | 非流式     | clearervoice | 语音分离   | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-ss-16k-onnx "modelscope")      |
| mossformer2-ss-8k-onnx     | 非流式     | clearervoice | 语音分离   | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-ss-8k-onnx "modelscope")       |
| frcrn-se-16k-onnx          | 非流式     | clearervoice | 语音降噪   | [modelscope](https://modelscope.cn/models/manyeyes/frcrn-se-16k-onnx "modelscope")            |
| uvr-kuielab_a_vocals-onnx  | 非流式     | uvr          | 语音分离   | [modelscope](https://modelscope.cn/models/manyeyes/uvr-kuielab_a_vocals-onnx "modelscope")    |
| uvr-kuielab_a_drums-onnx   | 非流式     | uvr          | 语音分离   | [modelscope](https://modelscope.cn/models/manyeyes/uvr-kuielab_a_drums-onnx "modelscope")     |


## 如何使用

暂时请参考示例代码进行集成与调用。


## 引用参考

1. [deezer/spleeter](https://github.com/deezer/spleeter)

2. [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)

3. [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

4. [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)