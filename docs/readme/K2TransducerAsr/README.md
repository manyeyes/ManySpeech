# ManySpeech.K2TransducerAsr User Guide

## I. Introduction
ManySpeech.K2TransducerAsr is a "speech recognition" library written in C#. Its underlying mechanism calls Microsoft.ML.OnnxRuntime to decode ONNX models. It has the following features:

### 1. Environmental Compatibility
It supports multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, which can meet the requirements of different development scenarios.

### 2. Cross-platform Compilation Features
It supports cross-platform compilation and can be used on platforms like Windows 7 SP1 or higher versions, macOS 10.13 (High Sierra) or higher versions, Linux distributions (specific dependencies are required, see the list of Linux distributions supported by.NET 6 for details), Android (Android 5.0 (API 21) or higher versions), and iOS.

### 3. Support for AOT Compilation
It is simple and convenient to use, facilitating developers to quickly integrate it into their projects.

## II. Installation Methods
It is recommended to install through the NuGet package manager. Here are two specific installation approaches:

### 1. Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.K2TransducerAsr
```

### 2. Using.NET CLI
Enter the following command in the command line to install:
```bash
dotnet add package ManySpeech.K2TransducerAsr
```

## III. Code Calling Methods

### 1. Offline (Non-streaming) Model Calling Method
#### 1.1 Adding Project References
```csharp
using ManySpeech.K2TransducerAsr;
using ManySpeech.K2TransducerAsr.Model;
```

#### 1.2 Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```

#### 1.3 Calling
```csharp
List<float[]> samples = new List<float[]>();
// The code for converting wav files to samples is omitted here...
// Refer to the examples in ManySpeech.K2TransducerAsr.Examples for details.

// Single recognition
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
    Console.WriteLine(result.text);
}

// Batch recognition
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

#### 1.4 Output Results
- **Single Recognition**:
```
after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds: 1062.28125
total_duration: 23340
rtf: 0.045513335475578405
```
- **Batch Recognition**:
```
after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds: 1268.6875
total_duration: 23340
rtf: 0.05435679091688089
```

### 2. Real-time (Streaming) Model Calling Method
#### 2.1 Adding Project References
```csharp
using ManySpeech.K2TransducerAsr;
using ManySpeech.K2TransducerAsr.Model;
```

#### 2.2 Model Initialization and Configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-multi-zh-hans-onnx-online-20231212";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```

#### 2.3 Calling
```csharp
List<List<float[]>> samplesList = new List<List<float[]>>();
// The code for converting wav files to samples is omitted here...
// The following is the sample code for batch processing:

// Batch processing
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
    //......(Some details are omitted here. Refer to the example code for details.)
	List<OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
	foreach (OnlineRecognizerResultEntity result in results_batch)
	{
		Console.WriteLine(result.text);
	}
	//......(Some details are omitted here. Refer to the example code for details.)
}

// Single processing
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
// Refer to the examples in ManySpeech.K2TransducerAsr.Examples for details.
```

#### 2.4 Output Results
- **Chinese Model Test Results**:
```
OnlineRecognizer:
batchSize: 1



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

elapsed_milliseconds: 2070.546875
total_duration: 9790
rtf: 0.21149610572012256
```
- **English Model Test Results**:
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

elapsed_milliseconds: 1088.890625
total_duration: 6625
rtf: 0.16436084905660378
```

## IV. Related Projects
- **Voice Endpoint Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library. Install it by using the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: To address the lack of punctuation in recognition results, you can add the ManySpeech.AliCTTransformerPunc library. Install it with the following command:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
Specific calling examples can refer to the official documentation of the corresponding libraries or the `ManySpeech.K2TransducerAsr.Examples` project. This project is a console/desktop example project, mainly used to demonstrate the basic functions of speech recognition, such as offline transcription and real-time recognition.

## V. Other Notes
- **Test Cases**: ManySpeech.K2TransducerAsr.Examples.
- **Test CPU**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz.

## VI. Model Downloads (Supported ONNX models)

| Model Name | Type | Supported Languages | Download Link |
|------------|------|---------------------|---------------|
| k2transducer-lstm-en-onnx-online-csukuangfj-20220903 | Streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-en-onnx-online-csukuangfj-20220903 "modelscope") |
| k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 | Streaming | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 "modelscope") |
| k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 | Streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 "modelscope") |
| k2transducer-zipformer-en-onnx-online-zengwei-20230517 | Streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-zengwei-20230517 "modelscope") |
| k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 | Streaming | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 "modelscope") |
| k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 | Streaming | Korean | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 "modelscope") |
| k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 | Streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 "modelscope") |
| k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 | Streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 "modelscope") |
| k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 | Streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
| k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 | Streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 "modelscope") |
| k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 | Streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
| k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 | Non-streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 "modelscope") |
| k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 | Non-streaming | Chinese | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 "modelscope") |
| k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 | Non-streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 "modelscope") |
| k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 | Non-streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 "modelscope") |
| k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 | Non-streaming | English | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 "modelscope") |
| k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 | Non-streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 "modelscope") |
| k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 | Non-streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 "modelscope") |
| k2transducer-zipformer-zh-en-onnx-offline-20231122 | Non-streaming | Chinese and English | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-en-onnx-offline-20231122 "modelscope") |
| k2transducer-zipformer-cantonese-onnx-offline-20240313 | Non-streaming | Cantonese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-cantonese-onnx-offline-20240313 "modelscope") |
| k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 | Non-streaming | Thai | [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 "modelscope") |
| k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 | Non-streaming | Japanese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 "modelscope") |
| k2transducer-zipformer-ru-onnx-offline-20240918 | Non-streaming | Russian | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ru-onnx-offline-20240918 "modelscope") |
| k2transducer-zipformer-vi-onnx-offline-20250420 | Non-streaming | Vietnamese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-vi-onnx-offline-20250420 "modelscope") |
| k2transducer-zipformer-ctc-zh-onnx-offline-20250703 | Non-streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-zh-onnx-offline-20250703 "modelscope")  |
| k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 | Non-streaming | Chinese | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 "modelscope") |

References
----------
[1] https://github.com/k2-fsa/icefall