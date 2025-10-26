 ( 简体中文 | [English](README.md) )

# ManySpeech.WhisperAsr 使用指南

## 一、简介
ManySpeech.WhisperAsr 是 [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech") 语音处理套件中一个专门的语音识别组件，支持 whisper,distil-whisper,whisper-turbo 等模型，其底层借助 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码，具备诸多优势：
- **多环境支持**：可兼容 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景的需求。
- **跨平台编译特性**：支持跨平台编译，无论是 Windows、macOS 还是 Linux、Android 等系统，都能进行编译使用，拓展了应用的范围。
- **支持 AOT 编译**：使用起来简单便捷，方便开发者快速集成到项目中。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下为两种具体安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.WhisperAsr
```

### （二）使用.NET CLI
在命令行中输入以下命令来安装：
```bash
dotnet add package ManySpeech.WhisperAsr
```

### （三）手动安装
在 NuGet 包管理器界面搜索「ManySpeech.WhisperAsr」，点击「安装」即可。

## 三、配置说明（参考：conf.json 文件）
用于解码的 conf.json 配置文件中，大部分参数无需改动，不过存在可修改的特定参数：
- `"task": "transcribe"`：使用 whisper 多语言模型（例如：whisper-tiny-onnx），指定 task 为 transcribe 时，只转写不翻译；为 translate 时，自动翻译为指定的语言类型；为空时，默认transcribe。
- `language: zh`：使用 whisper 多语言模型（例如：whisper-tiny-onnx），可指定语言类型，不指定时将自动识别语种。
- `without_timestamps: false`：为 false 时，识别结果有时间戳，为 true 时，无时间戳。

## 四、代码调用方法

### （一）离线（非流式）模型调用
1. **添加项目引用**
在代码中添加以下引用：
```csharp
using ManySpeech.WhisperAsr;
using ManySpeech.WhisperAsr.Model;
```
2. **模型初始化和配置**
    - **paraformer 模型初始化方式**：
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "whisper-tiny-onnx";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/conf.json";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, threadsNum: 1);
```
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，详细可参考 ManySpeech.WhisperAsr.Examples 示例代码
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

## 五、相关工程
- **语音端点检测**：为解决长音频合理切分问题，可添加 ManySpeech.AliFsmnVad 库，通过以下命令安装：
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **文本标点预测**：针对识别结果缺乏标点的情况，可添加 ManySpeech.AliCTTransformerPunc 库，安装命令如下：
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
具体的调用示例可参考对应库的官方文档或者 `ManySpeech.WhisperAsr.Examples` 项目。该项目是一个控制台/桌面端示例项目，主要用于展示语音识别的基础功能，像离线转写、实时识别等操作。

## 六、其他说明
- **测试用例**：以 `ManySpeech.WhisperAsr.Examples` 作为测试用例。
- **测试 CPU**：使用的测试 CPU 为 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz（2.59 GHz）。
- **支持平台**：
    - **Windows**：Windows 7 SP1 及更高版本。
    - **macOS**：macOS 10.13 (High Sierra) 及更高版本，也支持 ios 等。
    - **Linux**：适用于 Linux 发行版，但需要满足特定的依赖关系（详见.NET 6 支持的 Linux 发行版列表）。
    - **Android**：支持 Android 5.0 (API 21) 及更高版本。

## 七、模型下载（支持的 ONNX 模型）
以下是 ManySpeech.WhisperAsr 所支持的 ONNX 模型相关信息，包含模型名称、类型、支持语言、标点情况、时间戳情况以及下载地址等内容，方便根据具体需求选择合适的模型进行下载使用：

| 模型名称 | 类型 | 支持语言 | 标点 | 时间戳 | 下载地址 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| whisper-tiny-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-tiny-onnx "modelscope") |
| whisper-tiny-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-tiny-en-onnx "modelscope") |
| whisper-base-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-base-onnx "modelscope") |
| whisper-base-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-base-en-onnx "modelscope") |
| whisper-small-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-onnx "modelscope") |
| whisper-small-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-en-onnx "modelscope") |
| whisper-small-cantonese-onnx | 非流式 | 粤语、中文、英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-small-cantonese-onnx "modelscope") |
| whisper-medium-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-medium-onnx "modelscope") |
| whisper-medium-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-medium-en-onnx "modelscope") |
| whisper-large-v1-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v1-onnx "modelscope") |
| whisper-large-v2-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v2-onnx "modelscope") |
| whisper-large-v3-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-onnx "modelscope") |
| whisper-large-v3-turbo-onnx | 非流式 | 多语言 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-turbo-onnx "modelscope") |
| whisper-large-v3-turbo-zh-onnx | 非流式 | 中文、英文等 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/whisper-large-v3-turbo-zh-onnx "modelscope") |
| distil-whisper-small-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-small-en-onnx "modelscope") |
| distil-whisper-medium-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-medium-en-onnx "modelscope") |
| distil-whisper-large-v2-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v2-en-onnx "modelscope") |
| distil-whisper-large-v3-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v3-en-onnx "modelscope") |
| distil-whipser-large-v3.5-en-onnx | 非流式 | 英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whipser-large-v3.5-en-onnx "modelscope") |
| distil-whisper-large-v2-multi-hans-onnx | 非流式 | 中文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-large-v2-multi-hans-onnx "modelscope") |
| distil-whisper-small-cantonese-onnx-alvanlii-20240404 | 非流式 | 粤语、中文、英文 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/distil-whisper-small-cantonese-onnx-alvanlii-20240404 "modelscope") |


**引用参考**
[1] https://github.com/openai/whisper