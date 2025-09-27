# K2TransducerAsr

##### Introduction:

K2TransducerAsr is a "speech recognition" library written in C#. It calls Microsoft.ML.OnnxRuntime at the bottom layer to decode onnx models, supporting multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, supports cross-platform compilation, and supports AOT compilation. It is simple and convenient to use.

##### Supported Models (ONNX)
| Model Name  |  Type | Supported Language  | Download Address  |
| ------------ | ------------ | ------------ | ------------ |
|  k2transducer-lstm-en-onnx-online-csukuangfj-20220903 | Streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-en-onnx-online-csukuangfj-20220903 "modelscope") |
|  k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 | Streaming  | Chinese  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-lstm-zh-onnx-online-csukuangfj-20221014 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 | Streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-weijizhuang-20221202 "modelscope") |
|  k2transducer-zipformer-en-onnx-online-zengwei-20230517 | Streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-online-zengwei-20230517 "modelscope") |
|  k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 | Streaming  | Chinese  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-multi-zh-hans-onnx-online-20231212 "modelscope") |
|  k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 | Streaming  | Korean  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-ko-onnx-online-johnbamma-20240612 "modelscope") |
|  k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 | Streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-online-20250401 "modelscope") |
|  k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 | Streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 | Streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 | Streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 | Streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630 "modelscope") |
|  k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 | Non-streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-en-onnx-offline-csukuangfj-20220513 "modelscope") |
|  k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 | Non-streaming  | Chinese  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727 "modelscope") |
|  k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 | Non-streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-en-onnx-offline-yfyeung-20230417 "modelscope") |
|  k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 | Non-streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 | Non-streaming  | English  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516 "modelscope") |
|  k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 | Non-streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615 "modelscope") |
|  k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 | Non-streaming  | Chinese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902 "modelscope") |
|  k2transducer-zipformer-zh-en-onnx-offline-20231122 | Non-streaming  | Chinese and English  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-zh-en-onnx-offline-20231122 "modelscope") |
|  k2transducer-zipformer-cantonese-onnx-offline-20240313 | Non-streaming  | Cantonese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-cantonese-onnx-offline-20240313 "modelscope") |
|  k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 | Non-streaming  | Thai  |  [modelscope](https://www.modelscope.cn/models/manyeyes/k2transducer-zipformer-th-onnx-offline-yfyeung-20240620 "modelscope") |
|  k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 | Non-streaming  | Japanese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801 "modelscope") |
|  k2transducer-zipformer-ru-onnx-offline-20240918 | Non-streaming  | Russian  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ru-onnx-offline-20240918 "modelscope") |
|  k2transducer-zipformer-vi-onnx-offline-20250420 | Non-streaming  | Vietnamese  |  [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-vi-onnx-offline-20250420 "modelscope") |
|  k2transducer-zipformer-ctc-zh-onnx-offline-20250703 | Non-streaming  | Chinese  | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-zh-onnx-offline-20250703 "modelscope")  [github](https://github.moeyy.xyz/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2 "github") |
|  k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 | Non-streaming  | Chinese  | [modelscope](https://modelscope.cn/models/manyeyes/k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716 "modelscope") |


##### How to Use
###### 1. Clone the project source code
```bash
cd /path/to
git clone https://github.com/manyeyes/K2TransducerAsr.git
```
###### 2. Download the models from the above list to the directory: /path/to/K2TransducerAsr/K2TransducerAsr.Examples
```bash
cd /path/to/K2TransducerAsr/K2TransducerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[model name].git
```
###### 3. Load the project using vs2022 (or other IDEs)
###### 4. Set the files in the model directory to: Copy to Output Directory -> Copy if newer
###### 5. Modify the code in the example: string modelName = [model name]
###### 6. Run the project

## Calling Method for Offline (Non-streaming) Models:
###### 1. Add project reference
using K2TransducerAsr;
using K2TransducerAsr.Model;

###### 2. Model initialization and configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
K2TransducerAsr.OfflineRecognizer offlineRecognizer = new K2TransducerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
###### 3. Call
```csharp
List<float[]> samples = new List<float[]>();
//The code for converting wav files to samples is omitted here...
//Single recognition
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
    Console.WriteLine(result.text);
}
//Batch recognition
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
###### 4. Output results:
Single recognition
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1062.28125
total_duration:23340
rtf:0.045513335475578405
end!
```
Batch recognition
```
 after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

 god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonoured bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:1268.6875
total_duration:23340
rtf:0.05435679091688089
end!
```

## Calling Method for Real-time (Streaming) Models:

###### 1. Add project reference
using K2TransducerAsr;
using K2TransducerAsr.Model;

###### 2. Model initialization and configuration
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "k2transducer-zipformer-multi-zh-hans-onnx-online-20231212";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string joinerFilePath = applicationBase + "./" + modelName + "/joiner.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
K2TransducerAsr.OnlineRecognizer onlineRecognizer = new K2TransducerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: 2);
```
###### 3. Call
```csharp
List<List<float[]>> samplesList = new List<List<float[]>>();
//The code for converting wav files to samples is omitted here...
//The following is示意 code for batch processing:
//Batch processing
List<K2TransducerAsr.OnlineStream> onlineStreams = new List<K2TransducerAsr.OnlineStream>();
List<bool> isEndpoints = new List<bool>();
List<bool> isEnds = new List<bool>();
for (int num = 0; num < samplesList.Count; num++)
{
    K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    onlineStreams.Add(stream);
    isEndpoints.Add(false);
    isEnds.Add(false);
}
while (true)
{
    //......(Some details are omitted here, please refer to the example code for details)
	List<K2TransducerAsr.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
	foreach (K2TransducerAsr.OnlineRecognizerResultEntity result in results_batch)
	{
		Console.WriteLine(result.text);
	}
	//......(Some details are omitted here, please refer to the example code for details)
}
//Single processing
for (int j = 0; j < samplesList.Count; j++)
{
    K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    foreach (float[] samplesItem in samplesList[j])
    {
        stream.AddSamples(samplesItem);
        OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
        Console.WriteLine(result_on.text);
    }
}
//Please refer to the example (K2TransducerAsr.Examples) code for details
```

###### 4. Output results
* Test results of Chinese model:

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
end!
```

* Test results of English model:

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
end!
```

###### Related Projects:
* Voice activity detection to solve the problem of reasonable segmentation of long audio. Project address: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* Text punctuation prediction to solve the problem that the recognition results have no punctuation. Project address: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

###### Other Instructions:

Test case: K2TransducerAsr.Examples.
Test CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
Supported platforms:
Windows 7 SP1 or later,
macOS 10.13 (High Sierra) or later, iOS, etc.,
Linux distributions (specific dependencies are required, see the list of Linux distributions supported by .NET 6 for details),
Android (Android 5.0 (API 21) or later).

References
----------
[1] https://github.com/k2-fsa/icefall

[2] https://github.com/naudio/NAudio