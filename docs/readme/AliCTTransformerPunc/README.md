（[简体中文](README.zh_CN.md) |  English）

# ManySpeech.AliCTTransformerPunc
#### Introduction:

ManySpeech.AliCTTransformerPunc is a "text punctuation prediction" library written in C#. Under the hood, it utilizes Microsoft.ML.OnnxRuntime to decode ONNX models. This library boasts excellent compatibility in terms of framework adaptation, supporting multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+. It supports cross-platform compilation as well as AOT compilation, and is simple and convenient to use.

#### Supported Models (ONNX)
| Model Name | Vocabulary Size | Supported Languages | Download Link |
| --- | --- | --- | --- |
| alicttransformerpunc-zh-en-onnx | 272727 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-zh-en-onnx "modelscope") |
| alicttransformerpunc-large-zh-en-onnx | 471067 | Chinese, English | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-large-zh-en-onnx "modelscope") |

#### Punc Common Parameters (Reference: punc.yaml file):
The punc.yaml configuration parameters used for decoding generally do not need to be modified.

## Model Invocation Method:

###### 1. Add Project Reference
using ManySpeech.AliCTTransformerPunc;

###### 2. Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "alicttransformerpunc-large-zh-en-onnx";
string modelFilePath = applicationBase + "./"+ modelName + "/model.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/punc.yaml";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
```

###### 3. Invocation
```csharp
string text = "As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed when the fish dropped It is a big school of dolphin he thought They are widespread and the flying fish have little chance The bird has no chance The flying fish are too big for him and they go too fast 他正看着鸟儿又斜起翅 膀准备俯冲它向下冲来然后又猛烈地扇动着双翼追踪小飞鱼但是没有成效老人看见大海豚在追赶小飞鱼时海面微微隆起的水浪海豚在飞掠的鱼下面破水而行等鱼一落下海豚就会飞速潜人水中这群海豚真大呀他想它们分散开去小飞鱼很少有机会逃脱军舰鸟也没有机会小飞鱼对它来说太大了并且它们速度太快";
string result = ctTransformer.GetResults(text:text,splitSize:15);
```

###### 4. Output Results:
```
load_model_elapsed_milliseconds:979.125
As , he watched the bird dipped , again , slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish . The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish . The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed . when the fish dropped . It is a big school of dolphin . he thought They are widespread , and the flying fish have little chance . The bird has no chance . The flying fish are too big for him , and they go too fast . 他正看着鸟儿又斜起翅膀，准备俯冲，它向下冲来，然后又猛烈地扇动着双翼追踪小飞鱼，但是没有成效。老人看见大海豚在追赶 小飞鱼时，海面微微隆起的水浪，海豚在飞掠的鱼下面破水而行，等鱼一落下，海豚就会飞速。潜人水中，这群海豚真大呀，他想它们分散开去，小飞鱼很少有机会逃脱，军舰鸟也没有机会。小飞鱼对它来说太大了并且它们速度太快。
elapsed_milliseconds:381.6953125
end!
```

###### Related Projects:
* Speech Recognition, solving the problem of converting speech to text, project address: [AliParaformerAsr](https://github.com/manyeyes/AliParaformerAsr "AliParaformerAsr")
* Voice Endpoint Detection, solving the problem of reasonable segmentation of long audio, project address: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad")

#### Other Notes:
Test Cases: AliCTTransformerPunc.Examples.
Test Environment: Windows 11.

## Model Introduction:
#### Model Purpose
The Punc model used in the project is the Controllable Time-delay Transformer model open-sourced by Alibaba's Damo Academy. It can be used for punctuation prediction of the text output by the speech recognition model.

#### Model Structure:
Controllable Time-delay Transformer (CTTransformerPunc) is the punctuation module in the efficient post-processing framework proposed by the voice team of Damo Academy. This project is a Chinese general-purpose punctuation model, which can be applied to the punctuation prediction of text input and can also be used in the post-processing steps of speech recognition results to assist the speech recognition module in outputting readable text results.

![](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

The structure of the Controllable Time-delay Transformer is shown in the above figure, consisting of three parts: Embedding, Encoder, and Predictor. Embedding is the superposition of word vectors and positional vectors. The Encoder can adopt different network structures, such as self-attention, conformer, SAN-M, etc. The Predictor predicts the type of punctuation after each token.

In terms of model selection, the Transformer model with superior performance was adopted. While the Transformer model has good performance, its serialized input characteristics bring a larger delay to the system. Conventional Transformers can see all future information, causing punctuation to depend on distant future information. This gives users an adverse experience where punctuation keeps changing and the results are not fixed for a long time. Based on this issue, we innovatively proposed the controllable time-delay Transformer model (Controllable Time-Delay Transformer, CT-Transformer), which effectively controls the delay of punctuation without losing model performance.

#### More Detailed Information:
https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary
