 ([简体中文](README.zh_CN.md) | English )

# ManySpeech.TextPunc User Guide

## 1. Introduction
ManySpeech.TextPunc is a "text punctuation prediction" library written in C#. Its underlying layer leverages Microsoft.ML.OnnxRuntime to decode ONNX models. The library exhibits excellent features in multiple aspects:
- **Strong framework compatibility**: Compatible with net461+, net60+, netcoreapp3.1, netstandard2.0+ and other environments, meeting the usage requirements of different development scenarios.
- **Cross-platform compilation support**: Can be compiled on Windows, macOS, Linux, Android and other platforms, facilitating deployment in various operating system environments.
- **AOT compilation support**: Simple and convenient to use, easy for developers to integrate into corresponding projects.

## 2. Installation Methods
The following are several ways to install the library, with NuGet Package Manager recommended as the preferred method:

### (1) Using Package Manager Console
Execute the following command in Visual Studio's **Package Manager Console**:
```bash
Install-Package ManySpeech
```

### (2) Using .NET CLI
Enter the following command in the command line to complete the installation:
```bash
dotnet add package ManySpeech
```

### (3) Manual Installation
Open the NuGet Package Manager interface, search for **ManySpeech** in the search bar, and click the **Install** button to complete the installation.

## 3. Common Punc Parameters (Reference: punc.yaml File)
Parameters in the punc.yaml configuration file used for text punctuation prediction decoding generally do not need to be modified in actual use.

## 4. Model Calling Methods

### (1) Add Project Reference
Add the following reference statement to the code file:
```csharp
using ManySpeech.TextPunc;
```

### (2) Model Initialization and Configuration
Initialize and configure the model according to the following code example:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "alicttransformerpunc-large-zh-en-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string configFilePath = applicationDomain.BaseDirectory + "./" + modelName + "/punc.yaml";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
PuncRestorer puncRestorer = new PuncRestorer(modelFilePath, configFilePath, tokensFilePath);
```

### (3) Calling
The following is an example of calling code, using a piece of text containing Chinese and English as the input example:
```csharp
string text = "As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed when the fish dropped It is a big school of dolphin he thought They are widespread and the flying fish have little chance The bird has no chance The flying fish are too big for him and they go too fast 他正看着鸟儿又斜起翅 膀准备俯冲它向下冲来然后又猛烈地扇动着双翼追踪小飞鱼但是没有成效老人看见大海豚在追赶小飞鱼时海面微微隆起的水浪海豚在飞掠的鱼下面破水而行等鱼一落下海豚就会飞速潜人水中这群海豚真大呀他想它们分散开去小飞鱼很少有机会逃脱军舰鸟也没有机会小飞鱼对它来说太大了并且它们速度太快";
string result = puncRestorer.GetResults(text:text,splitSize:15);
```

### (4) Output Result Example
```
load_model_elapsed_milliseconds:979.125
As, he watched the bird dipped, again, slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish. The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish. The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed. when the fish dropped. It is a big school of dolphin. he thought They are widespread, and the flying fish have little chance. The bird has no chance. The flying fish are too big for him, and they go too fast. 他正看着鸟儿又斜起翅膀，准备俯冲，它向下冲来，然后又猛烈地扇动着双翼追踪小飞鱼，但是没有成效。老人看见大海豚在追赶 小飞鱼时，海面微微隆起的水浪，海豚在飞掠的鱼下面破水而行，等鱼一落下，海豚就会飞速。潜人水中，这群海豚真大呀，他想它们分散开去，小飞鱼很少有机会逃脱，军舰鸟也没有机会。小飞鱼对它来说太大了并且它们速度太快。
elapsed_milliseconds:381.6953125
end!
```

## 5. Related Projects
- **Speech Processing Suite**: Project address: [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech"). This suite covers the full-link models for speech processing, including speech recognition, voice activity detection, noise reduction and enhancement, and other functional modules, providing a comprehensive solution for speech-related processing.

## 6. Other Notes
- **Test Cases**: Use `ManySpeech.TextPunc.Examples` as test cases to facilitate users' reference and function verification.
- **Test Environment**: Tests were conducted in the Windows 11 environment, and users can perform test verification in similar environments according to actual conditions.

## 7. Model Download (Supported ONNX Models)
The following is information about the ONNX models supported by the library, including model name, vocabulary size, supported languages, and download address, to facilitate selecting appropriate models for download and use according to specific needs:

| Model Name | Vocabulary Size | Supported Languages | Download Address |
| ---- | ---- | ---- | ---- |
| alicttransformerpunc-zh-en-onnx | 272727 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-zh-en-onnx "modelscope") |
| alicttransformerpunc-large-zh-en-onnx | 471067 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-large-zh-en-onnx "modelscope") |
| fireredpunc-zh-en-onnx | 21128 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/fireredpunc-zh-en-onnx "modelscope") |

## 8. Model Introduction

### (1) CT-Transformer Model
#### Purpose
The CT-Transformer (Controllable Time-delay Transformer) model is an open-source model proposed by Alibaba DAMO Academy. Its core purpose is to perform punctuation prediction on the text output by speech recognition models to improve text readability. For more detailed model information, please visit the following link: https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary

#### Structure
Controllable Time-delay Transformer (CTTransformerPunc) is a punctuation module in the efficient post-processing framework proposed by the Speech Team of DAMO Academy, and it is a general Chinese punctuation model with the following specific structure:

![Controllable Time-delay Transformer Model Structure](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

The model mainly consists of three parts: Embedding, Encoder, and Predictor:
- **Embedding**: Superimposes word vectors and position vectors to provide basic vector representation for subsequent processing.
- **Encoder**: Can adopt different network structures, such as self-attention, conformer, SAN-M, etc., to extract and encode features of the input text.
- **Predictor**: Its main function is to predict the punctuation type after each token, thereby adding appropriate punctuation marks to the text.

In terms of model selection, the high-performance Transformer model is adopted because it can achieve good performance. However, due to its characteristics such as serialized input, the Transformer model brings significant latency to the system. Conventional Transformer models can see all future information, which causes the determination of punctuation to rely on distant future information, resulting in a poor user experience where punctuation keeps changing and refreshing, and the results are not fixed for a long time.

To solve this problem, the innovative Controllable Time-Delay Transformer (CT-Transformer) is proposed, which can effectively control the punctuation delay while ensuring no loss of model performance, making the punctuation prediction results more stable and reasonable.

### (2) FireRedPunc Model
FireRedPunc: Chinese and English punctuation prediction based on BERT. For more detailed model information, please visit the following link: https://modelscope.cn/models/xukaituo/FireRedPunc