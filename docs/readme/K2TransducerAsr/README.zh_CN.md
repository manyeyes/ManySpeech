# ManySpeech.K2TransducerAsr 使用指南

## 一、简介
ManySpeech.K2TransducerAsr 是一个采用 C# 编写的“语音识别”库，其底层通过调用 Microsoft.ML.OnnxRuntime 对 onnx 模型进行解码。具备以下特点：
- **环境兼容性**：支持 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景需求。
- **跨平台编译特性**：支持跨平台编译，可在 Windows 7 SP1 或更高版本、macOS 10.13（High Sierra）或更高版本、Linux 发行版（需特定依赖关系，详见.NET 6 支持的 Linux 发行版列表）、Android（Android 5.0（API 21）或更高版本）以及 iOS 等平台上使用。
- **支持 AOT 编译**：使用起来简单方便，便于开发者快速集成到项目中。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下是两种具体的安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.K2TransducerAsr
```

### （二）使用.NET CLI
在命令行中输入以下命令来安装：
```bash
dotnet add package ManySpeech.K2TransducerAsr
```

## 三、代码调用方法

### （一）离线（非流式）模型调用方法
1. **添加项目引用**
```csharp
using ManySpeech.K2TransducerAsr;
using ManySpeech.K2TransducerAsr.Model;
```
2. **模型初始化和配置**
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
3. **调用**
```csharp
List<float[]> samples = new List<float[]>();
//这里省略 wav 文件转 samples...
//具体参考示例（ManySpeech.K2TransducerAsr.Examples）代码

//单个识别
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
    Console.WriteLine(result.text);
}

//批量识别
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
foreach (OfflineRecognizerResultEntity result in results)
{
    Console.WriteLine(result.text);
}
```
4. **输出结果**
    - **单个识别**：
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1062.28125
total_duration:23340
rtf:0.045513335475578405
```
    - **批量识别**：
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1268.6875
total_duration:23340
rtf:0.05435679091688089
```

### （二）实时（流式）模型调用方法
1. **添加项目引用**
```csharp
using ManySpeech.K2TransducerAsr;
using ManySpeech.K2TransducerAsr.Model;
```
2. **模型初始化和配置**
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-multi-zh-hans-onnx-online-20231212";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
3. **调用**
```csharp
List<List<float[]>> samplesList = new List<List<float[]>>();
//这里省略 wav 文件转 samples...
//以下是批处理示意代码：

//批处理
List<OnlineStream> onlineStreams = new List<OnlineStream>();
List<bool> isEndpoints = new List<bool>();
List<bool> isEnds = new List<bool>();
for (int num = 0; num < samplesList.Count; num++)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    onlineStreams.add(stream);
    isEndpoints.add(false);
    isEnds.add(false);
}
while (true)
{
    //......(这里省略了一些细节,具体参看示例代码)
	List<OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
	foreach (OnlineRecognizerResultEntity result in results_batch)
	{
		Console.WriteLine(result.text);
	}
	//......(这里省略了一些细节,具体参看示例代码)
}

//单处理
for (int j = 0; j < samplesList.Count; j++)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    foreach (float[] samplesItem in samplesList[j])
    {
        stream.AddSamples(samplesItem);
        OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
        Console.WriteLine(result_on.text);
    }
}
//具体参考示例（ManySpeech.K2TransducerAsr.Examples）代码
```
4. **输出结果**
- **中文模型测试结果**：
```
OnlineRecognizer:
batchSize:1



这是
这是第一种
这是第一种第二
这是第一种第二种
这是第一种第二种叫
这是第一种第二种叫
这是第一种第二种叫
这是第一种第二种叫呃
这是第一种第二种叫呃与
这是第一种第二种叫呃与 always
这是第一种第二种叫呃与 always always
这是第一种第二种叫呃与 always always什么
这是第一种第二种叫呃与 always always什么意思
是
是不是
是不是
是不是平凡
是不是平凡的啊
是不是平凡的啊不认
是不是平凡的啊不认识
是不是平凡的啊不认识记下来
是不是平凡的啊不认识记下来 f
是不是平凡的啊不认识记下来 frequent
是不是平凡的啊不认识记下来 frequently
是不是平凡的啊不认识记下来 frequently频
是不是平凡的啊不认识记下来 frequently频繁
是不是平凡的啊不认识记下来 frequently频繁的
是不是平凡的啊不认识记下来 frequently频繁的
elapsed_milliseconds:2070.546875
total_duration:9790
rtf:0.21149610572012256
```
- **英文模型测试结果** ：

```




 after

 after early

 after early

 after early nightfa

 after early nightfall the ye

 after early nightfall the yellow la

 after early nightfall the yellow lamps

 after early nightfall the yellow lamps would light

 after early nightfall the yellow lamps would light up

 after early nightfall the yellow lamps would light up here

 after early nightfall the yellow lamps would light up here and

 after early nightfall the yellow lamps would light up here and there

 after early nightfall the yellow lamps would light up here and there the squa

 after early nightfall the yellow lamps would light up here and there the squalid

 after early nightfall the yellow lamps would light up here and there the squalid quar

 after early nightfall the yellow lamps would light up here and there the squalid quarter of

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the bro

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothel

 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

elapsed_milliseconds:1088.890625
total_duration:6625
rtf:0.16436084905660378
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
具体的调用示例可参考对应库的官方文档或者 `ManySpeech.K2TransducerAsr.Examples` 项目。该项目是一个控制台/桌面端示例项目，主要用于展示语音识别的基础功能，像离线转写、实时识别等操作。

## 五、其他说明
- **测试用例**：ManySpeech.K2TransducerAsr.Examples。
- **测试 CPU**：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz。

## 六、模型下载（支持的 ONNX 模型）
| 模型名称  |  类型 | 支持语言  | 下载地址  |
| ------------ | ------------ | ------------ | ------------ |
|  k2transducer-lstm-en-onnx-online-csukuangfj-20220903 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-en-onnx-online-csukuangfj-20220903 "modelscope") |
|  k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 | 流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-zengwei-20230517 | 流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-zengwei-20230517 "modelscope") |
|  k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 | 流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 "modelscope") |
|  k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 | 流式  | 韩文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 "modelscope") |
|  k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 | 流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 "modelscope") |
|  k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 | 流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 | 流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 | 流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 | 流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 "modelscope") |
|  k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 | 非流式  | 中文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 "modelscope") |
|  k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 "modelscope") |
|  k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 | 非流式  | 英文  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 | 非流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 "modelscope") |
|  k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 | 非流式  | 中文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 "modelscope") |
|  k2transducer-zipformer-zh-en-onnx-offline-20231122 | 非流式  | 中英文  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-en-onnx-offline-20231122 "modelscope") |
|  k2transducer-zipformer-cantonese-onnx-offline-20240313 | 非流式  | 粤语  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-cantonese-onnx-offline-20240313 "modelscope") |
|  k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 | 非流式  | 泰语  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 "modelscope") |
|  k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 | 非流式  | 日语  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 "modelscope") |
|  k2transducer-zipformer-ru-onnx-offline-20240918 | 非流式  | 俄语  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ru-onnx-offline-20240918 "modelscope") |
|  k2transducer-zipformer-vi-onnx-offline-20250420 | 非流式  | 越南语  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-vi-onnx-offline-20250420 "modelscope") |
|  k2transducer-zipformer-ctc-zh-onnx-offline-20250703 | 非流式  | 中文  | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-zh-onnx-offline-20250703 "modelscope")  [github](https://github.moeyy.xyz/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2 "github") |
|  k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 | 非流式  | 中文  | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 "modelscope") |

参考
----------
[1] https://github.com/k2-fsa/icefall
