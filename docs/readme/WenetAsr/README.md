# WenetAsr
A C# library for decoding the Wenet ASR ONNX model  


## Overview  
This is a C# library designed for decoding Wenet ASR (Automatic Speech Recognition) ONNX models. Written in C#, it supports multiple environments including **net461+**, **net60+**, **netcoreapp3.1**, and **netstandard2.0+**. It enables cross-platform compilation and AOT (Ahead-of-Time) compilation, with a simple and user-friendly API.  

You can quickly build multi-platform applications using MAUI or Uno frameworks.  


## Differences Between WenetAsr and WenetAsr2 in the Project  
### 1. Common Features  
- Identical functionality and invocation methods.  
- Both support decoding of **streaming** and **non-streaming** (offline) models.  


### 2. Key Differences  

| Library   | Module for Loading Streaming/Non-Streaming Models | Model & Extensibility                                                                 |
|-----------|---------------------------------------------------|---------------------------------------------------------------------------------------|
| WenetAsr  | Combined into one module                          | 1. Loads ONNX models officially exported by Wenet. <br> 2. Concise code structure.     |
| WenetAsr2 | Separate modules (one for streaming, one for non-streaming) | 1. Loads ONNX models officially exported by Wenet. <br> 2. Easy to extend: If your custom-exported streaming/non-streaming ONNX models have different configuration parameters, you can adjust each module independently without mutual interference. |

**Recommendation**: If you have no secondary development needs and want to directly use Wenet’s officially exported ONNX models, **WenetAsr** is preferred.  


## Supported Models (ONNX)  

| Model Name                                      | Type       | Supported Language | Download Link                                                                 |
|-------------------------------------------------|------------|--------------------|-------------------------------------------------------------------------------|
| wenet-u2pp-conformer-aishell-onnx-online-20210601  | Streaming  | Chinese            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-online-20210601 "modelscope") |
| wenet-u2pp-conformer-aishell-onnx-offline-20210601 | Offline    | Chinese            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-offline-20210601 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 | Streaming  | Chinese            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 | Offline    | Chinese            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 | Streaming  | English            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 | Offline    | English            | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 "modelscope") |


## Invocation Method for Offline (Non-Streaming) Models  
### 1. Add Project Reference  
```csharp
using WenetAsr;
```

### 2. Model Initialization and Configuration  
```csharp
// Load model files (get the base directory of the application)
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506";

// Paths to model components and token file
string encoderFilePath = Path.Combine(applicationBase, modelName, "encoder.int8.onnx");
string decoderFilePath = Path.Combine(applicationBase, modelName, "decoder.int8.onnx");
string ctcFilePath = Path.Combine(applicationBase, modelName, "ctc.int8.onnx");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

// Initialize the offline recognizer
WenetAsr.OfflineRecognizer offlineRecognizer = new WenetAsr.OfflineRecognizer(
    encoderFilePath, 
    decoderFilePath, 
    ctcFilePath, 
    tokensFilePath
);
```

### 3. Invocation  
```csharp
// Note: Audio file-to-sample conversion is omitted here. 
// For details, refer to `test_WenetAsrOfflineRecognizer` in the `examples` directory.

// Create an offline stream
WenetAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();

// Add audio samples to the stream
stream.AddSamples(sample);

// Get recognition results
WenetAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);

// Print the result
Console.WriteLine(result.Text);
```

### 4. Output Results  
#### Chinese (using model: wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506)  
```
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

啊这是第一种第二种叫嗯与欧维斯欧维斯什么意思

蒋永伯被拍到带着女儿出游

周望君就落实控物价

每当新年的钟声敲响的时候我总会闭起眼睛静静地许愿有时也会给自己定下新年的奋斗目标还有时听到新年的终身时我的心里会有一种遗憾的感觉感慨时光过的如此匆匆而自己往年的愿 望还没达成尽管如此经过岁月的洗礼我已长大成熟学会了勇敢的面对现实的一切

elapsed_milliseconds:6729.828125
total_duration:57944.0625
rtf:0.11614353282530027
end!
```

#### English (using model: wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728)  
```
after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:2639.1171875
total_duration:23340
rtf:0.11307271583119109
end!
```


## Invocation Method for Real-Time (Streaming) Models  
### 1. Add Project Reference  
```csharp
using WenetAsr;
```

### 2. Model Initialization and Configuration  
```csharp
// Load model files (get the base directory of the application)
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-online-20220506";

// Paths to model components and token file
string encoderFilePath = Path.Combine(applicationBase, modelName, "encoder.int8.onnx");
string decoderFilePath = Path.Combine(applicationBase, modelName, "decoder.int8.onnx");
string ctcFilePath = Path.Combine(applicationBase, modelName, "ctc.int8.onnx");
string tokensFilePath = Path.Combine(applicationBase, modelName, "tokens.txt");

// Initialize the online recognizer
WenetAsr.OnlineRecognizer onlineRecognizer = new WenetAsr.OnlineRecognizer(
    encoderFilePath, 
    decoderFilePath, 
    ctcFilePath, 
    tokensFilePath
);
```

### 3. Invocation  
```csharp
// Note: Audio file-to-sample conversion (or audio input from a microphone) is omitted here. 
// For details, refer to `test_WenetAsrOnlineRecognizer` in the `examples` directory.

// Create an online stream
WenetAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();

while (true)
{
    // Note: This is a simplified decoding example. 
    // For a complete and robust workflow, refer to the `examples` directory.
    
    // Get audio samples (from audio files or microphone)
    // sample = audio samples from input source
    
    // Add samples to the stream
    stream.AddSamples(sample);
    
    // Get real-time recognition results
    WenetAsr.Model.OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
    
    // Print the incremental result
    Console.WriteLine(result.Text);
}
```

### 4. Output Results  
```
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

elapsed_milliseconds:2148.8515625
total_duration:13052
rtf:0.16463772314587802
end!
```


## Related Projects  
1. **Voice Activity Detection (VAD)**  
   Solves the problem of reasonable segmentation of long audio.  
   Project URL: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad")  

2. **Text Punctuation Prediction**  
   Solves the problem of missing punctuation in recognition results.  
   Project URL: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")  


## Additional Notes  
- **Test Cases**: Refer to `WenetAsr.Examples` for complete usage examples.  
- **Test CPU**: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz  
- **Supported Platforms**:  
  - Windows 7 SP1 or later  
  - macOS 10.13 (High Sierra) or later, iOS  
  - Linux distributions (specific dependencies required; see the list of Linux distributions supported by .NET 6)  
  - Android (Android 5.0 (API 21) or later)  
- **Audio Sample Conversion**: The **NAudio library** is used for audio sample processing in the examples.  


## References  
[1] https://github.com/wenet-e2e/wenet  