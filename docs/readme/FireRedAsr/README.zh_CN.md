# ManySpeech.FireRedAsr 使用指南

## 一、简介
ManySpeech.FireRedAsr 是一款用于解码 FireRedASR AED-L 模型的 C# 库，专注于语音识别（ASR）任务。其底层借助 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码，具备诸多优势：
### （一）兼容性与框架支持
- **多环境支持**：可兼容 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景的需求。
- **跨平台编译特性**：支持跨平台编译，无论是 Windows、macOS 还是 Linux、Android、iOS 等系统，都能进行编译使用，拓展了应用的范围。
- **支持 AOT 编译**：使用起来简单便捷，方便开发者快速集成到项目中。

### （二）核心模型相关
其核心依赖的 FireRedASR-AED 模型，以平衡高性能与计算效率为设计目标，采用基于注意力机制的编码器 - 解码器（AED）架构，可作为基于大语言模型（LLM）的语音模型中的高效语音表示模块，为语音识别任务提供稳定且优质的技术支撑。

FireRedASR 本身是一系列支持普通话、中国方言和英语的开源工业级自动语音识别 (ASR) 模型，在公共普通话 ASR 基准测试中达到了新的最先进水平 (SOTA)，还具备出色的歌词识别能力。它包含两个变体：
- FireRedASR-LLM：旨在实现最先进的性能，并支持无缝端到端语音交互，采用编码器 - 适配器 - 大型语言模型 (LLM) 的框架。
- FireRedASR-AED：旨在平衡高性能与计算效率，并作为基于 LLM 的语音模型中的有效语音表示模块，利用基于注意力机制的编码器 - 解码器 (AED) 架构。而 fireredasr-aed-large-zh-en-onnx-offline-20250124 是从 FireRedASR-AED-L 导出的 onnx 模型，支持 one/batch 解码，可通过 manyspeech 在本地部署，实现语音转录。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下是几种具体的安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.FireRedAsr
```

### （二）使用.NET CLI
在命令行中执行：
```bash
dotnet add package ManySpeech.FireRedAsr
```

### （三）手动安装
在 NuGet 包管理器界面搜索「ManySpeech.FireRedAsr」，点击「安装」即可。

## 三、代码调用方法

### （一）离线（非流式）模型调用
1. **添加项目引用**
在代码中添加以下引用：
```csharp
using ManySpeech.FireRedAsr;
using ManySpeech.FireRedAsr.Model;
```
2. **模型初始化和配置**
**paraformer 模型初始化方式**：
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
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，详细可参考 ManySpeech.FireRedAsr.Examples 示例代码
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```
4. **输出结果示例**
```
朱立南在上市见面会上表示

这是第一种第二种叫呃与always always什么意思啊

好首先说一下刚才这个经理说完了这个销售问题咱再说一下咱们的商场问题首先咱们商场上半年业这个先各部门儿汇报一下就是业绩

elapsed_milliseconds:4391.234375
total_duration:21015.0625
rtf:0.2089565222563578
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

## 五、其他说明
- **测试用例**：FireRedASR.Examples。
- **测试 CPU**：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz。
- **支持的模型（ONNX）**：

| 模型名称  |  类型 |  支持语言  | 下载地址  |
| ------------ | ------------ | ------------ | ------------ |
|  fireredasr-aed-large-zh-en-onnx-offline-20250124 | 非流式  | 中文、英文  |[modelscope](https://www.modelscope.cn/models/manyeyes/fireredasr-aed-large-zh-en-onnx-offline-20250124 "modelscope") |

**引用参考**：
[1] https://github.com/FireRedTeam/FireRedASR 