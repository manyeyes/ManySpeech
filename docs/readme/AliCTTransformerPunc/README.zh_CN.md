（ 简体中文  | [英文](README.md) ）

# AliCTTransformerPunc
##### 简介：

AliCTTransformerPunc是一个使用C#编写的“文本标点预测”库，底层调用Microsoft.ML.OnnxRuntime对onnx模型进行解码。该类库在框架适配方面具有良好的兼容性，支持 net461+、net60+、netcoreapp3.1 及 netstandard2.0+ 等多种环境，支持跨平台编译，支持AOT编译。使用简单方便。

##### 支持的模型（ONNX）
|  模型名称 |词汇量|  支持语言 | 下载地址  |
| ------------ | ------------ | ------------ | ------------ |
| alicttransformerpunc-zh-en-onnx  |272727|  中文、英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-zh-en-onnx "modelscope")  |
|  alicttransformerpunc-large-zh-en-onnx |471067|  中文、英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/alicttransformerpunc-large-zh-en-onnx "modelscope")  |

##### Punc常用参数（参考：punc.yaml文件）：
用于解码的punc.yaml配置参数，通常在使用中无需修改。

## 模型调用方法：

###### 1.添加项目引用
using AliCTTransformerPunc;

###### 2.模型初始化和配置
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "alicttransformerpunc-large-zh-en-onnx";
string modelFilePath = applicationBase + "./"+ modelName + "/model.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/punc.yaml";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
AliCTTransformerPunc.CTTransformer ctTransformer = new CTTransformer(modelFilePath, configFilePath, tokensFilePath);
```
###### 3.调用
```csharp
string text = "As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed when the fish dropped It is a big school of dolphin he thought They are widespread and the flying fish have little chance The bird has no chance The flying fish are too big for him and they go too fast 他正看着鸟儿又斜起翅 膀准备俯冲它向下冲来然后又猛烈地扇动着双翼追踪小飞鱼但是没有成效老人看见大海豚在追赶小飞鱼时海面微微隆起的水浪海豚在飞掠的鱼下面破水而行等鱼一落下海豚就会飞速潜人水中这群海豚真大呀他想它们分散开去小飞鱼很少有机会逃脱军舰鸟也没有机会小飞鱼对它来说太大了并且它们速度太快";
string result = ctTransformer.GetResults(text:text,splitSize:15);
```
###### 4.输出结果：
```
load_model_elapsed_milliseconds:979.125
As , he watched the bird dipped , again , slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish . The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish . The dolphin were cutting through the water below the flight of the fish and would be in the water driving at speed . when the fish dropped . It is a big school of dolphin . he thought They are widespread , and the flying fish have little chance . The bird has no chance . The flying fish are too big for him , and they go too fast . 他正看着鸟儿又斜起翅膀，准备俯冲，它向下冲来，然后又猛烈地扇动着双翼追踪小飞鱼，但是没有成效。老人看见大海豚在追赶 小飞鱼时，海面微微隆起的水浪，海豚在飞掠的鱼下面破水而行，等鱼一落下，海豚就会飞速。潜人水中，这群海豚真大呀，他想它们分散开去，小飞鱼很少有机会逃脱，军舰鸟也没有机会。小飞鱼对它来说太大了并且它们速度太快。
elapsed_milliseconds:381.6953125
end!
```

###### 相关工程：
* 语音识别，解决语音转文本的问题，项目地址：[AliParaformerAsr](https://github.com/manyeyes/AliParaformerAsr "AliParaformerAsr") 
* 语音端点检测，解决长音频合理切分的问题，项目地址：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 

##### 其他说明：
测试用例：AliCTTransformerPunc.Examples。
测试环境：windows11。

## 模型介绍：
##### 模型用途
项目中使用的Punc模型是阿里巴巴达摩院开源的Controllable Time-delay Transformer模型。可用于语音识别模型输出文本的标点预测。

##### 模型结构：
Controllable Time-delay Transformer（CTTransformerPunc）是达摩院语音团队提出的高效后处理框架中的标点模块。本项目为中文通用标点模型，模型可以被应用于文本类输入的标点预测，也可应用于语音识别结果的后处理步骤，协助语音识别模块输出具有可读性的文本结果。

![](https://www.modelscope.cn/api/v1/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

Controllable Time-delay Transformer 模型结构如上图所示，由 Embedding、Encoder 和 Predictor 三部分组成。Embedding 是词向量叠加位置向量。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 预测每个token后的标点类型。

在模型的选择上采用了性能优越的Transformer模型。Transformer模型在获得良好性能的同时，由于模型自身序列化输入等特性，会给系统带来较大时延。常规的Transformer可以看到未来的全部信息，导致标点会依赖很远的未来信息。这会给用户带来一种标点一直在变化刷新，长时间结果不固定的不良感受。基于这一问题，我们创新性的提出了可控时延的Transformer模型（Controllable Time-Delay Transformer, CT-Transformer），在模型性能无损失的情况下，有效控制标点的延时。

##### 更详细的资料：
https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary
