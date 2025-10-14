 ( 简体中文 | [English](README.md) )

# ManySpeech.AliCTTransformerPunc 使用指南

## 一、简介
ManySpeech.AliCTTransformerPunc 是一款采用 C# 编写的“文本标点预测”库，其底层借助 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码操作。该库在多方面展现出良好特性：
- **框架适配性强**：能够兼容 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，可满足不同开发场景下的使用需求。
- **支持跨平台编译**：无论是 Windows、macOS 还是 Linux、Android 等平台，都可进行编译，方便在各类操作系统环境中部署应用。
- **支持 AOT 编译**：使用起来简单便捷，易于开发者上手集成到相应项目中。

## 二、安装方式
以下是几种安装该库的方式，推荐优先使用 NuGet 包管理器进行安装：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.AliCTTransformerPunc
```

### （二）使用.NET CLI
在命令行中输入以下命令完成安装：
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```

### （三）手动安装
打开 NuGet 包管理器界面，在搜索栏中输入「ManySpeech.AliCTTransformerPunc」，然后点击「安装」按钮即可完成安装操作。

## 三、Punc 常用参数（参考：punc.yaml 文件）
在对文本进行标点预测时，用于解码的 punc.yaml 配置文件里的参数，通常情况下在实际使用中无需进行修改。

## 四、模型调用方法

### （一）添加项目引用
在代码文件中添加如下引用语句：
```csharp
using ManySpeech.AliCTTransformerPunc;
```

### （二）模型初始化和配置
按照以下代码示例进行模型的初始化与配置操作：
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "alicttransformerpunc-large-zh-en-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string configFilePath = applicationDomain.BaseDirectory + "./" + modelName + "/punc.yaml";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
```

### （三）调用
示例调用代码如下，这里给出了一段包含中英文的文本作为输入示例：
```csharp
string text = "As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed when the fish dropped It is a big school of dolphin he thought They are widespread and the flying fish have little chance The bird has no chance The flying fish are too big for him and they go too fast 他正看着鸟儿又斜起翅 膀准备俯冲它向下冲来然后又猛烈地扇动着双翼追踪小飞鱼但是没有成效老人看见大海豚在追赶小飞鱼时海面微微隆起的水浪海豚在飞掠的鱼下面破水而行等鱼一落下海豚就会飞速潜人水中这群海豚真大呀他想它们分散开去小飞鱼很少有机会逃脱军舰鸟也没有机会小飞鱼对它来说太大了并且它们速度太快";
string result = ctTransformer.GetResults(text:text,splitSize:15);
```

### （四）输出结果示例
```
load_model_elapsed_milliseconds:979.125
As, he watched the bird dipped, again, slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish. The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish. The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed. when the fish dropped. It is a big school of dolphin. he thought They are widespread, and the flying fish have little chance. The bird has no chance. The flying fish are too big for him, and they go too fast. 他正看着鸟儿又斜起翅膀，准备俯冲，它向下冲来，然后又猛烈地扇动着双翼追踪小飞鱼，但是没有成效。老人看见大海豚在追赶 小飞鱼时，海面微微隆起的水浪，海豚在飞掠的鱼下面破水而行，等鱼一落下，海豚就会飞速。潜人水中，这群海豚真大呀，他想它们分散开去，小飞鱼很少有机会逃脱，军舰鸟也没有机会。小飞鱼对它来说太大了并且它们速度太快。
elapsed_milliseconds:381.6953125
end!
```

## 五、相关工程
- **语音处理套件**：项目地址为 [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech") ，该套件涵盖了语音处理全链路模型，包含语音识别、语音端点检测、降噪增强等多个功能模块，为语音相关处理提供了全面的解决方案。

## 六、其他说明
- **测试用例**：以 `ManySpeech.AliCTTransformerPunc.Examples` 作为测试用例，方便使用者参考和验证功能。
- **测试环境**：测试是在 windows11 环境下进行的，使用者可根据实际情况在类似环境中进行测试验证。

## 七、模型下载（支持的 ONNX 模型）
以下是该库支持的 ONNX 模型相关信息，包含模型名称、词汇量、支持语言以及下载地址等内容，方便根据具体需求选择合适的模型进行下载使用：

| 模型名称 | 词汇量 | 支持语言 | 下载地址 |
| ---- | ---- | ---- | ---- |
| alicttransformerpunc-zh-en-onnx | 272727 | 中文、英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-zh-en-onnx "modelscope") |
| alicttransformerpunc-large-zh-en-onnx | 471067 | 中文、英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-large-zh-en-onnx "modelscope") |

## 八、模型介绍

### （一）模型用途
本项目中所使用的 Punc 模型是由阿里巴巴达摩院开源的 Controllable Time-delay Transformer 模型，其核心用途在于针对语音识别模型输出的文本进行标点预测，提升文本的可读性。

### （二）模型结构
Controllable Time-delay Transformer（CTTransformerPunc）属于达摩院语音团队提出的高效后处理框架中的标点模块，是一个中文通用标点模型，具体结构如下：

![Controllable Time-delay Transformer 模型结构](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

该模型主要由 Embedding、Encoder 和 Predictor 三部分构成：
- **Embedding**：是将词向量与位置向量进行叠加，为后续处理提供基础的向量表示。
- **Encoder**：能够采用不同的网络结构，例如 self-attention、conformer、SAN-M 等，用于对输入文本的特征进行提取和编码。
- **Predictor**：主要功能是预测每个 token 后面的标点类型，以此为文本添加合适的标点符号。

在模型选择方面，之所以采用性能优越的 Transformer 模型，是因其能够获得良好的性能表现。然而，Transformer 模型由于自身具有序列化输入等特性，会给系统带来较大的时延问题。常规的 Transformer 模型能够看到未来的全部信息，这就导致标点的确定会依赖很远的未来信息，进而使得用户在查看结果时，会有一种标点一直在变化刷新、长时间结果不固定的不良感受。

为解决这一问题，创新性地提出了可控时延的 Transformer 模型（Controllable Time-Delay Transformer, CT-Transformer），它能够在保证模型性能不受损失的前提下，有效地控制标点的延时问题，使得标点预测结果更加稳定、合理。

### （三）更详细的资料
如需了解更详细的模型信息，可访问以下链接：
https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary