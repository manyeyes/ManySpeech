# ManySpeech.FireRedAsr User Guide

## I. Introduction
ManySpeech.FireRedAsr is a C# library used for decoding the FireRedASR AED-L model, focusing on the Automatic Speech Recognition (ASR) task. Its underlying mechanism utilizes Microsoft.ML.OnnxRuntime to decode ONNX models, and it has several advantages:

### (I) Compatibility and Framework Support
- **Multi-environment Support**: It is compatible with multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, which can meet the needs of different development scenarios.
- **Cross-platform Compilation Features**: It supports cross-platform compilation. Whether it's Windows, macOS, Linux, Android, iOS, or other systems, it can be compiled and used, expanding the scope of application.
- **Support for AOT Compilation**: It is simple and convenient to use, facilitating developers to quickly integrate it into their projects.

### (II) Core Model-related Aspects
The core FireRedASR-AED model, which it relies on, aims to balance high performance and computational efficiency. It adopts an Attention-based Encoder-Decoder (AED) architecture and can serve as an efficient speech representation module in speech models based on Large Language Models (LLMs), providing stable and high-quality technical support for speech recognition tasks.

FireRedASR itself is a series of open-source industrial-grade Automatic Speech Recognition (ASR) models that support Mandarin, Chinese dialects, and English. It has reached a new state-of-the-art (SOTA) level in public Mandarin ASR benchmark tests and also has excellent lyric recognition capabilities. It contains two variants:
- FireRedASR-LLM: Aimed at achieving the most advanced performance and supporting seamless end-to-end speech interaction, it adopts an Encoder-Adapter-Large Language Model (LLM) framework.
- FireRedASR-AED: Designed to balance high performance and computational efficiency, and serving as an effective speech representation module in LLM-based speech models, it utilizes an Attention-based Encoder-Decoder (AED) architecture. The fireredasr-aed-large-zh-en-onnx-offline-20250124 is an ONNX model derived from FireRedASR-AED-L, which supports one/batch decoding and can be deployed locally through ManySpeech to achieve speech transcription.

## II. Installation Methods
It is recommended to install via the NuGet package manager. Here are several specific installation approaches:

### (I) Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.FireRedAsr
```

### (II) Using.NET CLI
Execute the following command in the command line:
```bash
dotnet add package ManySpeech.FireRedAsr
```

### (III) Manual Installation
Search for "ManySpeech.FireRedAsr" in the NuGet package manager interface and click "Install".

## III. Code Calling Methods

### (I) Offline (Non-streaming) Model Calling
1. **Adding Project References**
Add the following references in the code:
```csharp
using ManySpeech.FireRedAsr;
using ManySpeech.FireRedAsr.Model;
```
2. **Model Initialization and Configuration**
**Paraformer Model Initialization Method**:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "fireredasr-aed-large-zh-en-onnx-offline-20250124";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/config.json";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, threadsNum: threadsNum);
```
3. **Calling Process**
```csharp
List<float[]> samples = new List<float[]>();
// Here, the relevant code for converting the wav file to samples is omitted. For details, please refer to the ManySpeech.FireRedAsr.Examples sample code.
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```
4. **Example of Output Results**
```
朱立南在上市见面会上表示

这是第一种第二种叫呃与always always什么意思啊

好首先说一下刚才这个经理说完了这个销售问题咱再说一下咱们的商场问题首先咱们商场上半年业这个先各部门儿汇报一下就是业绩

elapsed_milliseconds: 4391.234375
total_duration: 21015.0625
rtf: 0.2089565222563578
```

## IV. Related Projects
- **Voice Endpoint Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library. Install it using the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: To address the lack of punctuation in recognition results, you can add the ManySpeech.AliCTTransformerPunc library. Install it using the following command:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```

## V. Other Notes
- **Test Cases**: FireRedASR.Examples.
- **Test CPU**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz.
- **Supported Models (ONNX)**:

| Model Name | Type | Supported Languages | Download Address |
| ---- | ---- | ---- | ---- |
| fireredasr-aed-large-zh-en-onnx-offline-20250124 | Non-streaming | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/fireredasr-aed-large-zh-en-onnx-offline-20250124 "modelscope") |

**Reference**:
[1] https://github.com/FireRedTeam/FireRedASR 