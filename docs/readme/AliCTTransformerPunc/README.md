# ManySpeech.AliCTTransformerPunc User Guide

## I. Introduction
ManySpeech.AliCTTransformerPunc is a "text punctuation prediction" library written in C#, which decodes ONNX models by calling Microsoft.ML.OnnxRuntime at the bottom. This library has excellent compatibility in terms of framework adaptation, supporting multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+. It supports cross-platform compilation and AOT compilation, and is simple and convenient to use.

## II. Installation Methods
Install via NuGet package manager (recommended):

### 2.1 Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.AliCTTransformerPunc
```

### 2.2 Using .NET CLI
Execute the following command in the command line:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```

### 2.3 Manual Installation
Search for "ManySpeech.AliCTTransformerPunc" in the NuGet Package Manager interface and click "Install".


## III. Common Punc Parameters (Refer to punc.yaml File)
The punc.yaml configuration parameters used for decoding generally do not need to be modified during use.


## IV. Model Calling Methods

### 1. Add Project References
```csharp
using ManySpeech.AliCTTransformerPunc;
```

### 2. Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "alicttransformerpunc-large-zh-en-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string configFilePath = applicationDomain.BaseDirectory + "./" + modelName + "/punc.yaml";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
```

### 3. Calling the Method
```csharp
string text = "As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed when the fish dropped It is a big school of dolphin he thought They are widespread and the flying fish have little chance The bird has no chance The flying fish are too big for him and they go too fast 他正看着鸟儿又斜起翅 膀准备俯冲它向下冲来然后又猛烈地扇动着双翼追踪小飞鱼但是没有成效老人看见大海豚在追赶小飞鱼时海面微微隆起的水浪海豚在飞掠的鱼下面破水而行等鱼一落下海豚就会飞速潜人水中这群海豚真大呀他想它们分散开去小飞鱼很少有机会逃脱军舰鸟也没有机会小飞鱼对它来说太大了并且它们速度太快";
string result = ctTransformer.GetResults(text:text,splitSize:15);
```

### 4. Output Result:
```
load_model_elapsed_milliseconds:979.125
As, he watched the bird dipped, again, slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish. The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish. The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed. when the fish dropped. It is a big school of dolphin. he thought They are widespread, and the flying fish have little chance. The bird has no chance. The flying fish are too big for him, and they go too fast. 他正看着鸟儿又斜起翅膀，准备俯冲，它向下冲来，然后又猛烈地扇动着双翼追踪小飞鱼，但是没有成效。老人看见大海豚在追赶 小飞鱼时，海面微微隆起的水浪，海豚在飞掠的鱼下面破水而行，等鱼一落下，海豚就会飞速。潜人水中，这群海豚真大呀，他想它们分散开去，小飞鱼很少有机会逃脱，军舰鸟也没有机会。小飞鱼对它来说太大了并且它们速度太快。
elapsed_milliseconds:381.6953125
end!
```


## V. Related Projects
- Speech processing suite, project address: [ManySpeech](https://github.com/manyeyes/ManySpeech "ManySpeech")
  * Includes full-link speech processing models, such as speech recognition, voice activity detection, noise reduction and enhancement, etc.


## VI. Other Instructions
- Test case: ManySpeech.AliCTTransformerPunc.Examples
- Test environment: Windows 11


## VII. Model Download (Supported ONNX Models)

| Model Name | Vocabulary Size | Supported Languages | Download Address |
|------------|-----------------|---------------------|------------------|
| alicttransformerpunc-zh-en-onnx | 272727 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-zh-en-onnx "modelscope") |
| alicttransformerpunc-large-zh-en-onnx | 471067 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-large-zh-en-onnx "modelscope") |


## VIII. Model Introduction

### Model Purpose
The Punc model used in the project is the Controllable Time-delay Transformer model open-sourced by Alibaba DAMO Academy. It can be used for punctuation prediction of text output by speech recognition models.

### Model Structure
Controllable Time-delay Transformer (CTTransformerPunc) is the punctuation module in the efficient post-processing framework proposed by the speech team of DAMO Academy. This project is a general Chinese punctuation model, which can be applied to punctuation prediction of text input and also to the post-processing step of speech recognition results, helping the speech recognition module output readable text results.

![Model Structure](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

The structure of the Controllable Time-delay Transformer model, as shown above, consists of three parts: Embedding, Encoder, and Predictor. Embedding is the superposition of word vectors and position vectors. The Encoder can adopt different network structures, such as self-attention, conformer, SAN-M, etc. The Predictor predicts the punctuation type after each token.

In terms of model selection, the high-performance Transformer model is adopted. While achieving good performance, the Transformer model will bring large delay to the system due to its own characteristics such as serialized input. The conventional Transformer can see all future information, resulting in punctuation depending on far future information. This will give users a bad experience that punctuation is constantly changing and refreshing, and the results are not fixed for a long time. To solve this problem, we innovatively propose a Controllable Time-Delay Transformer (CT-Transformer) model, which can effectively control the delay of punctuation without losing model performance.

### More Detailed Information
https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary