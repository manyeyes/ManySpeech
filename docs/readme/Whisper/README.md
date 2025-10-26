 ([¼òÌåÖÐÎÄ](README.zh_CN.md) | English )

# ManySpeech.WhisperAsr User Guide


## I. Introduction
ManySpeech.WhisperAsr is a specialized speech recognition component in the [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech") speech processing suite. It supports models such as whisper, distil-whisper, and whisper-turbo. Under the hood, it uses Microsoft.ML.OnnxRuntime for decoding ONNX models, offering several advantages:
- **Multi-environment support**: Compatible with net461+, net60+, netcoreapp3.1, and netstandard2.0+, adapting to various development scenarios.
- **Cross-platform compilation**: Supports cross-platform compilation for systems like Windows, macOS, Linux, and Android, expanding application scope.
- **AOT compilation support**: Easy to use, facilitating quick integration into projects.


## II. Installation Methods
It is recommended to install via the NuGet package manager. Here are two specific installation approaches:

### (I) Using Package Manager Console
Execute the following command in Visual Studio's "Package Manager Console":
```bash
Install-Package ManySpeech.WhisperAsr
```

### (II) Using .NET CLI
Enter the following command in the command line to install:
```bash
dotnet add package ManySpeech.WhisperAsr
```

### (III) Manual Installation
Search for "ManySpeech.WhisperAsr" in the NuGet Package Manager interface and click "Install".


## III. Configuration Instructions (Reference: conf.json File)
Most parameters in the conf.json configuration file for decoding do not need modification, but specific parameters can be adjusted:
- `"task": "transcribe"`: When using a Whisper multilingual model (e.g., whisper-tiny-onnx), setting `task` to `transcribe` enables transcription only (no translation); setting it to `translate` enables automatic translation to the specified language; if left empty, `transcribe` is used by default.
- `language: zh`: When using a Whisper multilingual model (e.g., whisper-tiny-onnx), you can specify the language type. If not specified, the language will be automatically recognized.
- `without_timestamps: false`: When `false`, recognition results include timestamps; when `true`, timestamps are excluded.


## IV. Code Calling Methods

### (I) Offline (Non-streaming) Model Calling
1. **Add Project References**
Add the following references in your code:
```csharp
using ManySpeech.WhisperAsr;
using ManySpeech.WhisperAsr.Model;
```

2. **Model Initialization and Configuration**
    - **Paraformer model initialization method**:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "whisper-tiny-onnx";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/conf.json";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, threadsNum: 1);
```

3. **Calling Process**
```csharp
List<float[]> samples = new List<float[]>();
// Code for converting WAV files to samples is omitted here. For details, refer to the ManySpeech.WhisperAsr.Examples sample code.
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
foreach (OfflineRecognizerResultEntity result in results_batch)
{
    Console.WriteLine(result.Text);
}
```


## V. Related Projects
- **Voice Activity Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library. Install it using the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: For recognition results lacking punctuation, add the ManySpeech.AliCTTransformerPunc library. Install it with:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```

For specific calling examples, refer to the official documentation of the corresponding library or the `ManySpeech.WhisperAsr.Examples` project. This project is a console/desktop sample project that demonstrates basic speech recognition functions such as offline transcription and real-time recognition.


## VI. Other Instructions
- **Test Cases**: Use `ManySpeech.WhisperAsr.Examples` as test cases.
- **Test CPU**: The test CPU used is Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (2.59 GHz).
- **Supported Platforms**:
  - **Windows**: Windows 7 SP1 and later versions.
  - **macOS**: macOS 10.13 (High Sierra) and later versions, including iOS.
  - **Linux**: Compatible with Linux distributions, but specific dependencies must be met (see the list of Linux distributions supported by .NET 6).
  - **Android**: Android 5.0 (API 21) and later versions.


## VII. Model Downloads (Supported ONNX Models)
The following is information about ONNX models supported by ManySpeech.WhisperAsr, including model names, types, supported languages, punctuation support, timestamp support, and download links. Choose the appropriate model based on your needs:

| Model Name | Type | Supported Languages | Punctuation | Timestamp | Download Link |
| ---- | ---- | ---- | ---- | ---- | ---- |
| whisper-tiny-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-tiny-onnx "modelscope") |
| whisper-tiny-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-tiny-en-onnx "modelscope") |
| whisper-base-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-base-onnx "modelscope") |
| whisper-base-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-base-en-onnx "modelscope") |
| whisper-small-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-onnx "modelscope") |
| whisper-small-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-en-onnx "modelscope") |
| whisper-small-cantonese-onnx | Non-streaming | Cantonese, Chinese, English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-cantonese-onnx "modelscope") |
| whisper-medium-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-medium-onnx "modelscope") |
| whisper-medium-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-medium-en-onnx "modelscope") |
| whisper-large-v1-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v1-onnx "modelscope") |
| whisper-large-v2-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v2-onnx "modelscope") |
| whisper-large-v3-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-onnx "modelscope") |
| whisper-large-v3-turbo-onnx | Non-streaming | Multilingual | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-turbo-onnx "modelscope") |
| whisper-large-v3-turbo-zh-onnx | Non-streaming | Chinese, English, etc. | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-turbo-zh-onnx "modelscope") |
| distil-whisper-small-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-small-en-onnx "modelscope") |
| distil-whisper-medium-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-medium-en-onnx "modelscope") |
| distil-whisper-large-v2-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v2-en-onnx "modelscope") |
| distil-whisper-large-v3-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v3-en-onnx "modelscope") |
| distil-whipser-large-v3.5-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whipser-large-v3.5-en-onnx "modelscope") |
| distil-whisper-large-v2-multi-hans-onnx | Non-streaming | Chinese | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v2-multi-hans-onnx "modelscope") |
| distil-whisper-small-cantonese-onnx-alvanlii-20240404 | Non-streaming | Cantonese, Chinese, English | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-small-cantonese-onnx-alvanlii-20240404 "modelscope") |


**Reference**  
[1] https://github.com/openai/whisper