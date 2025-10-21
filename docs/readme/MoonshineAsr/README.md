 ([简体中文](README.zh_CN.md) | English )

# ManySpeech.MoonshineAsr User Guide

## I. Introduction
ManySpeech.MoonshineAsr is a speech recognition component within the [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech") speech processing suite, specifically designed for inference with the Moonshine model. It is developed using C# and calls Microsoft.ML.OnnxRuntime at the underlying level to decode ONNX models. It has the following features:

### 1. Environmental Compatibility
It supports multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, which can meet the requirements of different development scenarios.

### 2. Cross-platform Compilation Features
It supports cross-platform compilation and can be used on platforms like Windows 7 SP1 or higher versions, macOS 10.13 (High Sierra) or higher versions (also supports iOS, etc.), Linux distributions (specific dependencies are required, see the list of Linux distributions supported by.NET 6 for details), and Android 5.0 (API 21) or higher versions.

### 3. Support for AOT Compilation
It is simple and convenient to use, facilitating developers to quickly integrate it into their projects.

## II. Installation Methods
It is recommended to install through the NuGet package manager. Here are two specific installation approaches:

### 1. Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.MoonshineAsr
```

### 2. Using.NET CLI
Enter the following command in the command line to install:
```bash
dotnet add package ManySpeech.MoonshineAsr
```

### 3. Example Project Introduction
#### 1. Usage Examples
Load the project using vs2022 (or other IDEs) and run ManySpeech.MoonshineAsr.Examples. There are three usage methods as follows:
```bash
// Three usage methods
// 1. Directly recognize a single audio file at a time (it is recommended to use smaller files for faster recognition)
test_MoonshineAsrOfflineRecognizer();
// 2. Recognize by inputting in segments, which is suitable for external VAD
test_MoonshineAsrOnlineRecognizer();
// 3. Recognize with streaming input, using the built-in VAD function for automatic sentence segmentation, which is more convenient
test_MoonshineAsrOnlineVadRecognizer();
```

#### 2. Operations Related to the Built-in VAD Function (if used)
If you use the built-in VAD function for streaming recognition, you also need to download the VAD model. The operation is as follows:
```bash
// Download the VAD model
cd /path/to/MoonshineAsr/MoonshineAsr.Examples
git clone https://www.modelscope.cn/manyeyes/alifsmnvad-onnx.git
```

## III. Code Calling Methods

### 1. Offline (Non-streaming) Model Calling
#### 1.1 Adding Project References
Add the following references in the code:
```csharp
using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;
```

#### 1.2 Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "moonshine-base-en-onnx";
string preprocessFilePath = applicationBase + "./" + modelName + "/preprocess.int8.onnx";
string encodeFilePath = applicationBase + "./" + modelName + "/encode.int8.onnx";
string cachedDecodeFilePath = applicationBase + "./" + modelName + "/cached_decode.int8.onnx";
string uncachedDecodeFilePath = applicationBase + "./" + modelName + "/uncached_decode.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/conf.json";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
```

#### 1.3 Calling Process
```csharp
List<float[]> samples = new List<float[]>();
// The code for converting wav files to samples is omitted here. For details, please refer to the examples in ManySpeech.MoonshineAsr.Examples.
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

### 2. Using Streaming Input to Call the Model for Recognition
#### 2.1 Adding Project References
Add the following references in the code as well:
```csharp
using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;
```

#### 2.2 Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string preprocessFilePath = applicationBase + "./" + modelName + "/preprocess.onnx";
string encodeFilePath = applicationBase + "./" + modelName + "/encode.onnx";
string cachedDecodeFilePath = applicationBase + "./" + modelName + "/cached_decode.onnx";
string uncachedDecodeFilePath = applicationBase + "./" + modelName + "/uncached_decode.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
string vadModelFilePath = applicationBase + "/" + vadModelName + "/" + "model.int8.onnx";
string vadMvnFilePath = applicationBase + vadModelName + "/" + "vad.mvn";
string vadConfigFilePath = applicationBase + vadModelName + "/" + "vad.json";
OnlineVadRecognizer onlineVadRecognizer = new OnlineVadRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, vadModelFilePath, vadConfigFilePath, vadMvnFilePath, threadsNum: 1);
```

#### 2.3 Calling Process
```csharp
List<float[]> samples = new List<float[]>();
// The code for converting wav files to samples is omitted here. The following is sample code for batch processing:
List<OnlineVadStream> streams = new List<OnlineVadStream>();
foreach (var sample in samples)
{
    OnlineVadStream stream = onlineVadRecognizer.CreateOnlineVadStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineVadRecognizer.GetResults(streams);
// Single processing example, only need to build one stream
OnlineVadStream stream = onlineVadRecognizer.CreateOnlineVadStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineVadRecognizer.GetResult(stream);
// For details, please refer to the examples in ManySpeech.MoonshineAsr.Examples.
```

#### 2.4 Example of Recognition Results (with Timestamps) When Using Streaming Input for Recognition
```
[00:00:00,630-->00:00:06,790]
  thank you. Thank you.

[00:00:07,300-->00:00:10,760]
 Thank you everybody. All right, everybody go ahead and have a seat.

[00:00:11,450-->00:00:15,820]
 How's everybody doing today?

[00:00:17,060-->00:00:20,780]
 How about Tim Spicer?

[00:00:24,270-->00:00:30,450]
  I am here with students at Wakefield High School in Arlington, Virginia.

[00:00:31,070-->00:00:40,430]
 And we've got students tuning in from all across America from kindergarten through 12th grade. And I am just so glad

[00:00:40,960-->00:00:48,430]
 that all could join us today and I want to thank Wakefield for being such an outstanding host give yourselves a big round of applause

  // ...... (The following is omitted)
```

## IV. Related Projects
- **Voice Endpoint Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library. Install it by using the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: To address the lack of punctuation in recognition results, you can add the ManySpeech.AliCTTransformerPunc library. Install it with the following command:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
Specific calling examples can refer to the official documentation of the corresponding libraries or the `ManySpeech.MoonshineAsr.Examples` project. This project is a console/desktop example project, mainly used to demonstrate the basic functions of speech recognition, such as offline transcription and real-time recognition.

## V. Other Notes
- **Test Cases**: Use `ManySpeech.MoonshineAsr.Examples` as test cases.
- **Test CPU**: The test CPU used is Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (2.59 GHz).

## VI. Model Downloads (Supported ONNX Models)
| Model Name | Type | Supported Languages | Punctuation | Timestamp | Download Link |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  moonshine-base-en-onnx | Non-streaming | English | Yes | No |  [modelscope](https://modelscope.cn/models/manyeyes/moonshine-base-en-onnx "modelscope") |
|  moonshine-tiny-en-onnx | Non-streaming | English | Yes | No | [modelscope](https://modelscope.cn/models/manyeyes/moonshine-tiny-en-onnx "modelscope") |

## VII. Model Introduction
1. **Differences in Model Positioning**
Both models are English ASR models in the Moonshine series. The main differences lie in parameter scale and performance:
- `moonshine-tiny-en-onnx`: It is a lightweight model (with 27M parameters, approximately 190MB), suitable for resource-constrained devices (such as edge devices and embedded devices), balancing speed and basic recognition accuracy.
- `moonshine-base-en-onnx`: It is a basic-level model (with 62M parameters, approximately 400MB), with higher recognition accuracy than the Tiny version, suitable for scenarios where higher accuracy is required and hardware resources are relatively sufficient.

2. **Model Download Methods**
You can directly clone the model files using the Git command (you need to install the Git tool first). Taking `moonshine-tiny-en-onnx` as an example:
```bash
git clone https://www.modelscope.cn/manyeyes/moonshine-tiny-en-onnx.git
```

3. **Adaptation Scenarios**
Both models can support **offline (non-streaming) speech recognition** through the ManySpeech.MoonshineAsr library, and can also implement **real-time (streaming) recognition** by combining with built-in or external Voice Activity Detection (VAD) modules (such as ManySpeech.AliFsmnVad). They are applicable to scenarios such as speech transcription and real-time captioning.

**References**:
[1] https://github.com/usefulsensors/moonshine 