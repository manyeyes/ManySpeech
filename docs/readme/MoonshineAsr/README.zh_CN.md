 ( 简体中文 | [English](README.md) )

# ManySpeech.MoonshineAsr 使用指南

## 一、简介
ManySpeech.MoonshineAsr 是一个采用 C# 编写的“语音识别”库，其底层调用 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码。具备以下特点：
- **环境兼容性**：支持 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景需求。
- **跨平台编译特性**：支持跨平台编译，可应用于 Windows 7 SP1 及更高版本、macOS 10.13（High Sierra）及更高版本（也支持 iOS 等）、Linux 发行版（需满足特定依赖关系，详见.NET 6 支持的 Linux 发行版列表）以及 Android 5.0（API 21）及更高版本等平台。
- **支持 AOT 编译**：使用起来简单方便，便于开发者快速集成到项目中。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下是两种具体的安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.MoonshineAsr
```

### （二）使用.NET CLI
在命令行中输入以下命令来安装：
```bash
dotnet add package ManySpeech.MoonshineAsr
```

### （三）示例项目介绍
1. **使用方式示例**
使用 vs2022（或其他 IDE）加载项目，并运行 ManySpeech.MoonshineAsr.Examples，有以下三种使用方式：
```bash
// 三种使用方式
// 1.直接一次识别单个音频文件（建议文件小一点识别更快）
test_MoonshineAsrOfflineRecognizer();
// 2.分片输入识别，适用于外接 vad
test_MoonshineAsrOnlineRecognizer();
// 3.流式输入识别，使用内置 vad 功能，自动断句，更加便捷
test_MoonshineAsrOnlineVadRecognizer();
```
2. **内置 vad 功能相关操作（若使用）**
如果使用内置 vad 功能进行流式识别，还需下载 vad 模型，操作如下：
```bash
// 下载 vad 模型
cd /path/to/MoonshineAsr/MoonshineAsr.Examples
git clone https://www.modelscope.cn/manyeyes/alifsmnvad-onnx.git
```

## 三、代码调用方法

### （一）离线（非流式）模型调用
1. **添加项目引用**
在代码中添加以下引用：
```csharp
using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;
```
2. **模型初始化和配置**：
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
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，详细可参考 ManySpeech.MoonshineAsr.Examples 示例代码
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

### （二）使用流式输入的方式调用模型进行识别
1. **添加项目引用**
同样在代码中添加以下引用：
```csharp
using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;
```
2. **模型初始化和配置**
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
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，以下是批处理示意代码：
List<OnlineVadStream> streams = new List<OnlineVadStream>();
foreach (var sample in samples)
{
    OnlineVadStream stream = onlineVadRecognizer.CreateOnlineVadStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineVadRecognizer.GetResults(streams);
//单处理示例，只需构建一个 stream
OnlineVadStream stream = onlineVadRecognizer.CreateOnlineVadStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineVadRecognizer.GetResult(stream);
//具体可参考 ManySpeech.MoonshineAsr.Examples 示例代码
```

4. **使用流式输入的方式识别，识别结果（自带时间戳）示例**
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

  // ...... (以下省略)
```

## 四、相关工程
- **语音端点检测**：为解决长音频合理切分问题，可添加 ManySpeech.AliFsmnVad 库，通过以下命令安装：
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **文本标点预测**：针对识别结果缺乏标点的情况，可添加 ManySpeech.AliCTTransformerPunc 库，安装命令如下：
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
具体的调用示例可参考对应库的官方文档或者 `ManySpeech.MoonshineAsr.Examples` 项目。该项目是一个控制台/桌面端示例项目，主要用于展示语音识别的基础功能，像离线转写、实时识别等操作。

## 五、其他说明
- **测试用例**：以 `ManySpeech.MoonshineAsr.Examples` 作为测试用例。
- **测试 CPU**：使用的测试 CPU 为 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz（2.59 GHz）。

## 六、模型下载（支持的 ONNX 模型）
| 模型名称  |  类型 | 支持语言  | 标点  |  时间戳 | 下载地址  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  moonshine-base-en-onnx | 非流式  | 英文  |  是 | 否  |  [modelscope](https://modelscope.cn/models/manyeyes/moonshine-base-en-onnx "modelscope") |
|  moonshine-tiny-en-onnx | 非流式  | 英文  |  是 | 否  | [modelscope](https://modelscope.cn/models/manyeyes/moonshine-tiny-en-onnx "modelscope") |

## 七、模型介绍：
1. **模型定位差异**  
   - 两个模型均为 Moonshine 系列的英文 ASR 模型，区别主要在于参数规模和性能：
     - `moonshine-tiny-en-onnx`：轻量级模型（27M 参数，约 190MB），适合资源受限的设备（如边缘设备、嵌入式设备），兼顾速度与基础识别精度。
     - `moonshine-base-en-onnx`：基础级模型（62M 参数，约 400MB），识别精度高于 Tiny 版本，适合对精度要求稍高、硬件资源较充足的场景。

2. **模型下载方式**  
   可通过 Git 命令直接克隆模型文件（需先安装 Git 工具），以 `moonshine-tiny-en-onnx` 为例：
   ```bash
   git clone https://www.modelscope.cn/manyeyes/moonshine-tiny-en-onnx.git
   ```

3. **适配场景**  
   两个模型均支持通过 ManySpeech.MoonshineAsr 库实现 **离线（非流式）语音识别**，也可结合内置或外接的语音端点检测（VAD）模块（如 ManySpeech.AliFsmnVad）实现 **实时（流式）识别**，适用于语音转写、实时字幕等场景。

**引用参考**：

[1] https://github.com/usefulsensors/moonshine 