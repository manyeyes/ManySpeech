# AudioSep
a high-performance audio processing library built with C# , specializing in audio separation, noise reduction, and enhancement.It supports multiple environments including **net461+**, **net60+**, **netcoreapp3.1**, and **netstandard2.0+**, enabling cross-platform compilation and AOT (Ahead-of-Time) compilation. It is simple and easy to use. 

##### Supported Models (ONNX)

| Model Name  | Type | Reference  | Function | Download Link  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| ZipEnhancer-se-16k-base-onnx | Non-streaming | clearervoice | Speech Denoising | [modelscope](https://modelscope.cn/models/manyeyes/ZipEnhancer-se-16k-base-onnx "modelscope") |
| spleeter-2stems-44k-onnx | Non-streaming | spleeter | Speech Separation | [modelscope](https://modelscope.cn/models/manyeyes/spleeter-2stems-44k-onnx "modelscope") |
| gtcrn-se-16k-onnx-offline | Non-streaming | gtcrn | Speech Denoising | [modelscope](https://modelscope.cn/models/manyeyes/gtcrn-se-16k-onnx-offline "modelscope") |
| mossformer2-se-48k-onnx | Non-streaming | clearervoice | Speech Denoising | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-se-48k-onnx "modelscope") |
| mossformer2-sr-48k-onnx | Non-streaming | clearervoice | Speech Enhancement | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-sr-48k-onnx "modelscope") |
| mossformer2-ss-16k-onnx | Non-streaming | clearervoice | Speech Separation | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-ss-16k-onnx "modelscope") |
| mossformer2-ss-8k-onnx | Non-streaming | clearervoice | Speech Separation | [modelscope](https://modelscope.cn/models/manyeyes/mossformer2-ss-8k-onnx "modelscope") |
| frcrn-se-16k-onnx | Non-streaming | clearervoice | Speech Denoising | [modelscope](https://modelscope.cn/models/manyeyes/frcrn-se-16k-onnx "modelscope") |
| uvr-kuielab_a_vocals-onnx | Non-streaming | uvr | Speech Separation | [modelscope](https://modelscope.cn/models/manyeyes/uvr-kuielab_a_vocals-onnx "modelscope") |
| uvr-kuielab_a_drums-onnx | Non-streaming | uvr | Speech Separation | [modelscope](https://modelscope.cn/models/manyeyes/uvr-kuielab_a_drums-onnx "modelscope") |

##### How to Use

For now, please refer to the sample code.

Reference
----------
1. [deezer/spleeter](https://github.com/deezer/spleeter)
1. 
2. [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
1. 
3. [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
1. 
4. [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)