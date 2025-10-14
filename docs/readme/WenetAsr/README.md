# ManySpeech.WenetAsr User Guide

## I. Introduction
ManySpeech.WenetAsr is a C# library for decoding the Wenet ASR ONNX model, with the following features:
- **Environmental Compatibility**: Supports multiple environments including net461+, net60+, netcoreapp3.1, and netstandard2.0+.
- **Cross-platform Capability**: Supports cross-platform compilation and can run on systems such as Windows, macOS, Linux, Android, and iOS.
- **AOT Support**: Supports AOT compilation, ensuring simple and convenient usage.
- **Multi-platform Application Development**: Can be combined with MAUI or Uno to quickly build multi-platform applications.

## II. Installation Methods
### (I) Installation via NuGet Package Manager (Recommended)
1. **Using Package Manager Console**
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.WenetAsr
```
2. **Using .NET CLI**
Run the following command in the command line:
```bash
dotnet add package ManySpeech.WenetAsr
```
3. **Manual Installation**
Search for "ManySpeech.WenetAsr" in the NuGet Package Manager interface and click "Install".

### (II) Manual Reference via XML Nodes (For Scenarios Requiring Manual Project Configuration Management)
If you need to add references by directly modifying the project configuration file (e.g., `.csproj` file) through XML nodes, follow these steps:
1. Locate the `.csproj` file (e.g., `YourProject.csproj`) in the project root directory and open it with a text editor or IDE.
2. Add the following XML reference nodes within the `<ItemGroup>` node under the root `<Project>` node (if there is no `<ItemGroup>` node in the project, create it manually):
```xml
<ItemGroup>
  <!-- Add ManySpeech.WenetAsr reference; replace "Version" with the latest version number, which can be found on the NuGet official website -->
  <PackageReference Include="ManySpeech.WenetAsr" Version="x.x.x" />
</ItemGroup>
```
3. After saving the `.csproj` file, restart the IDE or execute the `dotnet restore` command in the command line to trigger dependency package restoration and complete the reference addition.

> Note: The `Version` attribute must be filled with the actual required version number (e.g., `1.0.0`). You can check the latest version on the [NuGet official website](https://www.nuget.org/packages/ManySpeech.WenetAsr) to ensure compatibility with the project's .NET environment.


## III. Differences Between WenetAsr and WenetAsr2
### (I) Common Features
- Same functionality: Both support decoding of streaming and non-streaming models.
- Consistent calling methods.

### (II) Differences

| Library Name | Streaming & Non-streaming Model Loading Module | Model & Extension |
| ---- | ---- | ---- |
| ManySpeech.WenetAsr | Integrated into one | 1. Loads ONNX models officially exported by Wenet<br>2. Concise code |
| ManySpeech.WenetAsr2 | Independent modules | 1. Loads ONNX models officially exported by Wenet<br>2. Easy to extend; if parameters of custom-exported streaming and non-streaming models differ, adjustments can be made in their respective modules without mutual interference |

**Recommendation**: If there is no need for secondary development and you want to directly use ONNX models officially exported by Wenet, WenetAsr is recommended.

## IV. Code Calling Methods
### (I) Offline (Non-streaming) Model Calling Method
1. **Add Project References**
```csharp
using ManySpeech.WenetAsr;
using ManySpeech.WenetAsr.Model;
```
2. **Model Initialization and Configuration**
```csharp
// Load model
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
```
3. **Calling Process**
```csharp
// The conversion from audio files to samples is omitted here; for details, refer to test_WenetAsrOfflineRecognizer in examples
OfflineStream stream = offlineRecognizer.CreateOfflineStream();
stream.AddSamples(sample);
Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
Console.WriteLine(result.Text);
```
4. **Output Result Examples**
- **Chinese Model** (wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506)
```
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

啊这是第一种第二种叫嗯与欧维斯欧维斯什么意思

蒋永伯被拍到带着女儿出游

周望君就落实控物价

每当新年的钟声敲响的时候我总会闭起眼睛静静地许愿有时也会给自己定下新年的奋斗目标还有时听到新年的终身时我的心里会有一种遗憾的感觉感慨时光过的如此匆匆而自己往年的愿 望还没达成尽管如此经过岁月的洗礼我已长大成熟学会了勇敢的面对现实的一切

elapsed_milliseconds:6729.828125
total_duration:57944.0625
rtf:0.11614353282530027
```
- **English Model** (wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728)
```
after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:2639.1171875
total_duration:23340
rtf:0.11307271583119109
```

### (II) Real-time (Streaming) Model Calling Method
1. **Add Project References**
```csharp
using ManySpeech.WenetAsr;
using ManySpeech.WenetAsr.Model;
```
2. **Model Initialization and Configuration**
```csharp
// Load model
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-online-20220506";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
```
3. **Calling Process**
```csharp
// The conversion from audio files to samples is omitted here, or for processing audio from a microphone
// For specific implementation, refer to the test_WenetAsrOnlineRecognizer sample code in ManySpeech.WenetAsr.Examples
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
while (true)
{
    // This is a simple decoding demonstration; for a more detailed process, refer to examples
    // sample = audio data from audio files or microphone
    stream.AddSamples(sample);
    OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
    Console.WriteLine(result.Text);
}
```
4. **Output Result Example**
```



正是因为

正是因为存在绝

正是因为存在绝对正义

正是因为存在绝对正义所以我们

正是因为存在绝对正义所以我们接受现

正是因为存在绝对正义所以我们接受现实的相对

正是因为存在绝对正义所以我们接受现实的相对正议

正是因为存在绝对正义所以我们接受现实的相对正议但

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

elapsed_milliseconds:2148.8515625
total_duration:13052
rtf:0.16463772314587802
```

## V. Related Projects
- **Voice Activity Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library. Install it using the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: To address the lack of punctuation in recognition results, you can add the ManySpeech.AliCTTransformerPunc library. Install it using the following command:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
For specific calling examples, refer to the official documentation of the corresponding libraries or the `ManySpeech.WenetAsr.Examples` project. This project is a console/desktop sample project that demonstrates basic speech recognition functions such as offline transcription and real-time recognition.

## VI. Additional Notes
- **Test Cases**: `ManySpeech.WenetAsr.Examples` is used as the test case.
- **Test CPU**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (2.59 GHz).
- **Supported Platforms**:
  - **Windows**: Windows 7 SP1 or later versions.
  - **macOS**: macOS 10.13 (High Sierra) or later versions, also supports iOS.
  - **Linux**: Applicable to Linux distributions (specific dependencies are required; see the list of Linux distributions supported by .NET 6 for details).
  - **Android**: Android 5.0 (API 21) or later versions.

## VII. Model Downloads (Supported ONNX Models)
| Model Name | Type | Supported Language | Download Link |
| ---- | ---- | ---- | ---- |
| wenet-u2pp-conformer-aishell-onnx-online-20210601 | Streaming | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-online-20210601 "modelscope") |
| wenet-u2pp-conformer-aishell-onnx-offline-20210601 | Offline | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-offline-20210601 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 | Streaming | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 | Offline | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 | Streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 | Offline | English | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 "modelscope") |

**References**:
[1] https://github.com/wenet-e2e/wenet