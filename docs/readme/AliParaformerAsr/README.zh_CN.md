 ( 简体中文 | [English](README.md) )

# ManySpeech.AliParaformerAsr 使用指南

## 一、简介
ManySpeech.AliParaformerAsr 是 [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech") 语音处理套件中一个专门的语音识别组件，支持paraformer-large、paraformer-seaco-large、sensevoice-small 等模型，其底层借助 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码，具备诸多优势：
- **多环境支持**：可兼容 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景的需求。
- **跨平台编译特性**：支持跨平台编译，无论是 Windows、macOS 还是 Linux、Android 等系统，都能进行编译使用，拓展了应用的范围。
- **支持 AOT 编译**：使用起来简单便捷，方便开发者快速集成到项目中。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下为两种具体安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.AliParaformerAsr
```

### （二）使用.NET CLI
在命令行中输入以下命令来安装：
```bash
dotnet add package ManySpeech.AliParaformerAsr
```

### （三）手动安装
在 NuGet 包管理器界面搜索「ManySpeech.AliParaformerAsr」，点击「安装」即可。

## 三、配置说明（参考：asr.yaml 文件）
用于解码的 asr.yaml 配置文件中，大部分参数无需改动，不过存在可修改的特定参数：
- `use_itn: true`：在使用 sensevoicesmall 模型配置时开启此参数，即可实现逆文本正则化功能，例如可将类似“123”这样的文本转换为“一百二十三”，让识别结果的文本表达更符合常规阅读习惯。

## 四、代码调用方法

### （一）离线（非流式）模型调用
1. **添加项目引用**
在代码中添加以下引用：
```csharp
using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
```
2. **模型初始化和配置**
    - **paraformer 模型初始化方式**：
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model_quant.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
    - **SeACo-paraformer 模型初始化方式**：
        - 首先，需在模型目录下找到 hotword.txt 文件，并按照每行一个中文词汇的格式添加自定义热词，例如添加行业术语、特定人名等热词内容。
        - 然后，在代码中新增相关参数，示例如下：
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string modelebFilePath = applicationBase + "./" + modelName + "/model_eb.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string hotwordFilePath = applicationBase + "./" + modelName + "/hotword.txt";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath);
```
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，详细可参考 ManySpeech.AliParaformerAsr.Examples 示例代码
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
欢迎大家来体验达摩院推出的语音识别模型

非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人

he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉

elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
```

### （二）实时（流式）模型调用
1. **添加项目引用**
同样在代码中添加以下引用：
```csharp
using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
```
2. **模型初始化和配置**
```csharp
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
3. **调用过程**
```csharp
List<float[]> samples = new List<float[]>();
//此处省略将 wav 文件转换为 samples 的相关代码，以下是批处理示意代码：
List<OnlineStream> streams = new List<OnlineStream>();
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
foreach (var sample in samples)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
//单处理示例，只需构建一个 stream
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
//具体可参考 ManySpeech.AliParaformerAsr.Examples 示例代码
```
4. **输出结果示例**
```

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
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
具体的调用示例可参考对应库的官方文档或者 `ManySpeech.AliParaformerAsr.Examples` 项目。该项目是一个控制台/桌面端示例项目，主要用于展示语音识别的基础功能，像离线转写、实时识别等操作。

## 六、其他说明
- **测试用例**：以 `ManySpeech.AliParaformerAsr.Examples` 作为测试用例。
- **测试 CPU**：使用的测试 CPU 为 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz（2.59 GHz）。
- **支持平台**：
    - **Windows**：Windows 7 SP1 及更高版本。
    - **macOS**：macOS 10.13 (High Sierra) 及更高版本，也支持 ios 等。
    - **Linux**：适用于 Linux 发行版，但需要满足特定的依赖关系（详见.NET 6 支持的 Linux 发行版列表）。
    - **Android**：支持 Android 5.0 (API 21) 及更高版本。

## 七、模型下载（支持的 ONNX 模型）
以下是 ManySpeech.AliParaformerAsr 所支持的 ONNX 模型相关信息，包含模型名称、类型、支持语言、标点情况、时间戳情况以及下载地址等内容，方便根据具体需求选择合适的模型进行下载使用：

| 模型名称 | 类型 | 支持语言 | 标点 | 时间戳 | 下载地址 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| paraformer-large-zh-en-onnx-offline | 非流式 | 中文、英文 | 否 | 否 | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"), [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-timestamp-onnx-offline | 非流式 | 中文、英文 | 否 | 是 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline "modelscope") |
| paraformer-large-en-onnx-offline | 非流式 | 英文 | 否 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-onnx-online | 流式 | 中文、英文 | 否 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online "modelscope") |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | 非流式 | 中文、粤语、英文 | 否 | 是 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 | 非流式 | 中文、粤语、英文 | 否 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 | 流式 | 中文、粤语、英文 | 否 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
| paraformer-seaco-large-zh-timestamp-onnx-offline | 非流式 | 中文、热词 | 否 | 是 | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
| SenseVoiceSmall | 非流式 | 中文、粤语、英文、日语、韩语 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |
| sensevoice-small-wenetspeech-yue-int8-onnx | 非流式 | 粤语、中文、英文、日语、韩语 | 是 | 否 | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx "modelscope") |

## 八、模型介绍

### （一）模型用途
Paraformer 是由达摩院语音团队提出的一种高效的非自回归端到端语音识别框架，本项目中的 Paraformer 中文通用语音识别模型采用工业级数万小时的标注音频进行训练，这使得模型具备良好的通用识别效果，可广泛应用于语音输入法、语音导航、智能会议纪要等多种场景，且有着较高的识别准确率。

### （二）模型结构
Paraformer 模型结构主要由 Encoder、Predictor、Sampler、Decoder 以及 Loss function 这五部分构成，其结构示意图可查看[此处](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)，各部分具体功能如下：
- **Encoder**：它可以采用不同的网络结构，像 self-attention、conformer、SAN-M 等，主要负责提取音频中的声学特征。
- **Predictor**：是一个两层的 FFN（前馈神经网络），其作用在于预测目标文字的个数，并且抽取目标文字对应的声学向量，为后续的识别处理提供关键数据。
- **Sampler**：属于无可学习参数模块，它能够依据输入的声学向量和目标向量，生成含有语义的特征向量，以此来丰富识别的语义信息。
- **Decoder**：结构与自回归模型类似，但它是双向建模（自回归模型为单向建模），通过双向的结构能够更好地对上下文进行建模，提升语音识别的准确性。
- **Loss function**：除了包含交叉熵（CE）与 MWER（最小词错误率）这两个区分性优化目标外，还涵盖了 Predictor 优化目标 MAE（平均绝对误差），通过这些优化目标来保障模型的精度。

### （三）主要核心点
- **Predictor 模块**：基于 Continuous integrate-and-fire (CIF) 的预测器（Predictor）来抽取目标文字对应的声学特征向量，借助这种方式能够更为精准地预测语音中目标文字的个数，提高语音识别的准确性。
- **Sampler**：通过采样操作，将声学特征向量与目标文字向量变换为含有语义信息的特征向量，然后与双向的 Decoder 配合，能够显著增强模型对于上下文的理解和建模能力，使识别结果更符合语义逻辑。
- **基于负样本采样的 MWER 训练准则**：这一训练准则有助于模型在训练过程中更好地优化参数，减少识别错误，提升整体的识别性能。

### （四）更详细的资料
- **模型链接**：
    - [paraformer-large-offline（非流式）](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline（非流式）")
    - [paraformer-large-online（流式）](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online（流式）")
    - [SenseVoiceSmall（非流式）](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall（非流式）")
- **论文**： [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")
- **论文解读**：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型")

**引用参考**
[1] https://github.com/alibaba-damo-academy/FunASR 