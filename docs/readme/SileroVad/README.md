 ([简体中文](README.md) | English )

# SileroVad

A C# voice activity detection library for speech activity detection based on the Silero VAD model.


## Introduction

**SileroVad** is a Voice Activity Detection (VAD) library developed in C#, with ONNX model decoding implemented under the hood using `Microsoft.ML.OnnxRuntime`. This library features the following:
- Multi-framework support: Compatible with .NET Framework 4.6.1+, .NET 6.0+, .NET Core 3.1, and .NET Standard 2.0+
- Cross-platform capability: Supports Windows, macOS, Linux, and other systems, enabling cross-platform compilation
- Flexible deployment: Supports AOT (Ahead-of-Time) compilation and is simple and convenient to use


## Supported Models (ONNX)

| Model Name            | Type              | Download Link                                                          |
|-----------------------|-------------------|-------------------------------------------------------------------------|
| silero-vad-v6-onnx    | Streaming/Offline | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v6-onnx)   |
| silero-vad-v5-onnx    | Streaming/Offline | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v5-onnx)   |
| silero-vad-onnx       | Streaming/Offline | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-onnx)      |


## Quick Start

### 1. Clone the Project Source Code
```bash
cd /path/to/your/workspace
git clone https://github.com/manyeyes/SileroVad.git
```

### 2. Download Model Files
Download the model from the table above to the sample project directory:
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
# Replace [Model Name] with the actual model name (e.g., silero-vad-v6-onnx)
git clone https://www.modelscope.cn/manyeyes/[Model Name].git
```

### 3. Configure the Project
- Load the solution using Visual Studio 2022 or another compatible IDE
- Set all files in the model directory to: **Copy to Output Directory -> Copy if newer**

### 4. Prepare Speech Recognition Model (Optional)
The sample implements speech recognition via the `OfflineRecognizer` method, which requires an additional ASR (Automatic Speech Recognition) model download:
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
git clone https://www.modelscope.cn/manyeyes/aliparaformerasr-large-zh-en-timestamp-onnx-offline.git
```

### 5. Run the Sample
- Modify the model path in the sample code: `string modelName = "[Model Directory Name]"`
- Program entry: `Program.cs`, which executes two test cases by default:
  - Offline detection: `TestOfflineVad()` (Source code: `OfflineVad.cs`)
  - Streaming detection: `TestOnlineVad()` (Source code: `OnlineVad.cs`)


## Running Results

### Offline Detection Output
```bash
load vad model elapsed_milliseconds:337.1796875
vad infer result:
00:00:00,000-->00:00:02,410
试错的过程很简单

00:00:02,934-->00:00:05,834
啊今特别是今天冒名插修卡的同学你们可以

00:00:05,974-->00:00:10,442
听到后面的有专门的活动课他会大大

00:00:10,582-->00:00:15,626
降低你的思错成本其实你也可以不要来听课为什么你自己写嘛

00:00:16,182-->00:00:19,818
我先今天写五个点我就实试试验一下发现这五个点不行

00:00:20,182-->00:00:22,026
我再写五个点这是再不行

00:00:22,422-->00:00:25,770
那再写五个点嘛你总会所谓的

00:00:25,942-->00:00:28,906
活动大神和所谓的高手

00:00:29,078-->00:00:34,634
都是只有一个把所有的错所有的坑全给趟

00:00:34,902-->00:00:37,898
一辩留下正确的你就是所谓的大神

00:00:38,518-->00:00:43,338
明白吗所以说关于活动通过这一块我只送给你们四个字啊换位思考

00:00:43,830-->00:00:47,082
如果说你要想降低你的试错成本

00:00:47,606-->00:00:49,802
今天来这里你们就是对的

00:00:50,166-->00:00:52,234
因为有创新创需要搞这个机会

00:00:52,470-->00:00:56,202
所以说关于活动过于不过这个问题或者活动很难通过这个话题

00:00:57,430-->00:01:01,930
我真的要坐下来聊的话要聊一天但是我觉得我刚才说的四个字

00:01:02,102-->00:01:03,466
足够好谢谢

00:01:03,862-->00:01:09,162
好非常感谢那个三毛老师的回答啊三毛老师说我们在整个店铺的这个活动当中我们要去

00:01:09,398-->00:01:10,470
换位思考其实

elapsed_milliseconds:4450.8671875
total_duration:70470.625
rtf:0.06315918423456582
------------------------
```

### Streaming Detection Output
```bash
load vad model elapsed_milliseconds:75.21875
00:00:00,032-->00:00:05,632
嗯 on time 就要准时 in time 是及时交他总是准时交他的作业

------------------------------
00:00:06,016-->00:00:11,360
那用一般现在时是没有什么感情色彩的陈述一个事实

------------------------------
00:00:11,552-->00:00:15,424
下一句话为什么要用现在进行时他的意思并不是说

------------------------------
00:00:15,776-->00:00:21,696
说他现在正在教他的

------------------------------
elapsed_milliseconds:819.7734375
total_duration:17640
rtf:0.046472417091836735
------------------------
```


## Related Projects

- **Speech Recognition**: [AliParaformerAsr](https://github.com/manyeyes/AliParaformerAsr)
- **Voice Activity Detection (Long Audio Segmentation)**: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad)
- **Text Punctuation Prediction**: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc)


## System Requirements

- **Test Environment**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz
- **Supported Platforms**:
  - Windows: Windows 7 SP1 and above
  - macOS: macOS 10.13 (High Sierra) and above (including iOS)
  - Linux: Distributions compatible with .NET 6 (specific dependencies required)
  - Android: Android 5.0 (API 21) and above


## References
[1] [Silero VAD](https://github.com/snakers4/silero-vad)