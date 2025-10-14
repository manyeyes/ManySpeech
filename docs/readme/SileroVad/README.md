 ([简体中文](README.zh_CN_.md) | English )

# ManySpeech.SileroVad User Guide

## I. Introduction
**ManySpeech.SileroVad** is a Voice Activity Detection (VAD) library developed in C#. Its underlying implementation uses `Microsoft.ML.OnnxRuntime` to decode ONNX models. This library has the following features:

### (I) Multi-environment Support
It is compatible with environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, which can meet the needs of different development scenarios.

### (II) Cross-platform Compilation Feature
It supports cross-platform compilation. Whether it's Windows, macOS, Linux, Android or other systems, it can be compiled and used, expanding the scope of application.

### (III) Support for AOT Compilation
It is simple and convenient to use, facilitating developers to quickly integrate it into their projects.

## II. Installation Methods
It is recommended to install via the NuGet package manager. Here are three specific installation methods:

### (I) Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.SileroVad
```

### (II) Using.NET CLI
Enter the following command in the command line to install:
```bash
dotnet add package ManySpeech.SileroVad
```

### (III) Manual Installation
Search for "ManySpeech.SileroVad" in the NuGet package manager interface and click "Install".

## III. Quick Start

### (I) Download the VAD Model
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
# Replace [Model Name] with the actual model name (e.g., silero-vad-v6-onnx)
git clone https://www.modelscope.cn/manyeyes/[Model Name].git
```

### (II) Download the ASR Model (Optional)
In the example, speech recognition is achieved through the `OfflineRecognizer` method, and an additional ASR model needs to be downloaded:
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
git clone https://www.modelscope.cn/manyeyes/aliparaformerasr-large-zh-en-timestamp-onnx-offline.git
```

### (III) Configure the Project
1. **Load the solution using Visual Studio 2022 or other compatible IDEs**: Open the project solution with appropriate development tools to prepare for subsequent operations.
2. **Set all files in the model directory to "Copy to Output Directory -> Copy if Newer"**: Ensure that the model files can be correctly output with the project so that they can be called normally during runtime, guaranteeing that the project can obtain the required model resources smoothly.

### (IV) Import Namespaces
After installation, import the following namespaces at the beginning of the code file:
```csharp
using ManySpeech.SileroVad;
using ManySpeech.SileroVad.Model;
```

### (V) Initialize the Model and Configure
Three core files (model file, configuration file, and mean normalization file) need to be prepared in advance. When initializing, specify the file paths and batch decoding parameters:
```csharp
// Get the application root directory (to avoid hard-coded paths)
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "silero-vad-v6-onnx"; // Model folder name

// Concatenate the full paths of the model, configuration, and mean normalization files
string modelFilePath = applicationBase + "./" + modelName + "/silero_vad.onnx";
string configFilePath = applicationBase + "./" + modelName + "/vad.yaml";

int batchSize = 2; // Batch decoding size (adjust according to hardware performance, recommended to be between 1 and 4)

// Initialize the AliFsmnVad instance (load the model and configure parameters)
AliFsmnVad aliFsmnVad = new OfflineVad(modelFilePath, configFilePath: configFilePath, threshold: 0F, isDebug: false);
```

### (VI) Call the Core Methods
Choose different calling methods according to the size of the audio file. Small files are suitable for one-time processing, while large files are recommended to be processed in steps to reduce memory usage:
```csharp
// samples: Audio sample data (needs to be read in advance through libraries like NAudio, in the format of float[])
// batch stream decode
List<OfflineStream> streams = new List<OfflineStream>();
foreach (float[] samplesItem in samples)
{
    OfflineStream stream = offlineVad.CreateOfflineStream();
    stream.AddSamples(samplesItem);
    streams.Add(stream);
}

Console.WriteLine("vad infer result:");
List<SileroVad.Model.VadResultEntity> results = offlineVad.GetResults(streams);
foreach (SileroVad.Model.VadResultEntity result in results)
{
    foreach (var item in result.Segments.Zip(result.Waveforms))
    {
        Console.WriteLine(string.Format("{0}-->{1}", TimeSpan.FromMilliseconds(item.First.Start / 16).ToString(@"hh\:mm\:ss\,fff"), TimeSpan.FromMilliseconds(item.First.End / 16).ToString(@"hh\:mm\:ss\,fff")));
        // Use the ManySpeech.AliParaformerAsr library to recognize the segmented samples
        // OfflineRecognizer(new List<float[]>() { item.Second });
        Console.WriteLine("");
    }
}
```

### (VII) Run the Examples Directly to See the Effects
1. **Modify the model path in the example code**: Change `string modelName = "[Model Directory Name]"` to the actual corresponding model directory name to ensure that the program can accurately find and load the model.
2. **Program Entry**: The program entry is `Program.cs`, and it executes two test cases by default:
    - **Offline Detection**: `TestOfflineVad()` (source code: `OfflineVad.cs`).
    - **Online Detection**: `TestOnlineVad()` (source code: `OnlineVad.cs`).

## IV. Running Results

### (I) Offline Detection Output
```bash
load vad model elapsed_milliseconds:337.1796875
vad infer result:
00:00:00,000-->00:00:02,410
loading asr model elapsed_milliseconds:1320.9375
试错的过程很简单

00:00:02,934-->00:00:05,834
啊今特别是今天冒名插修卡的同学你们可以

00:00:05,974-->00:00:10,442
听到后面的有专门的活动课他会大大

00:00:10,582-->00:00:15,626
降低你的思错成本其实你也可以不要来听课为什么你自己写嘛

00:00:16,182-->00:00:19,818
我先今天写五个点我就实试试验一下发现这五个点不行

00:00:20,182-->00:00:22,026
我再写五个点这是再不行

00:00:22,422-->00:00:25,770
那再写五个点嘛你总会所谓的

00:00:25,942-->00:00:28,906
活动大神和所谓的高手

00:00:29,078-->00:00:34,634
都是只有一个把所有的错所有的坑全给趟

00:00:34,902-->00:00:37,898
一辩留下正确的你就是所谓的大神

00:00:38,518-->00:00:43,338
明白吗所以说关于活动通过这一块我只送给你们四个字啊换位思考

00:00:43,830-->00:00:47,082
如果说你要想降低你的试错成本

00:00:47,606-->00:00:49,802
今天来这里你们就是对的

00:00:50,166-->00:00:52,234
因为有创新创需要搞这个机会

00:00:52,470-->00:00:56,202
所以说关于活动过于不过这个问题或者活动很难通过这个话题

00:00:57,430-->00:01:01,930
我真的要坐下来聊的话要聊一天但是我觉得我刚才说的四个字

00:01:02,102-->00:01:03,466
足够好谢谢

00:01:03,862-->00:01:09,162
好非常感谢那个三毛老师的回答啊三毛老师说我们在整个店铺的这个活动当中我们要去

00:01:09,398-->00:01:10,470
换位思考其实

elapsed_milliseconds:4450.8671875
total_duration:70470.625
rtf:0.06315918423456582
------------------------
```

### (II) Online Detection Output
```bash
load vad model elapsed_milliseconds:75.21875
00:00:00,032-->00:00:05,632
嗯 on time 就要准时 in time 是及时交他总是准时交他的作业

------------------------------
00:00:06,016-->00:00:11,360
那用一般现在时是没有什么感情色彩的陈述一个事实

------------------------------
00:00:11,552-->00:00:15,424
下一句话为什么要用现在进行时他的意思并不是说

------------------------------
00:00:15,776-->00:00:21,696
说他现在正在教他的

------------------------------
elapsed_milliseconds:819.7734375
total_duration:17640
rtf:0.046472417091836735
------------------------
```

## V. Related Projects
- **Speech Recognition**: To verify the effect of VAD, you can add the ManySpeech.AliParaformerAsr library to recognize the segmented samples. Install it using the following command:
```bash
dotnet add package ManySpeech.AliParaformerAsr
```

## VI. System Requirements

### (I) Test Environment
Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz.

### (II) Supported Platforms
- **Windows**: Windows 7 SP1 or later versions.
- **macOS**: macOS 10.13 (High Sierra) or later versions (including iOS).
- **Linux**: Compatible with the distributions supported by.NET 6 (specific dependencies need to be met).
- **Android**: Android 5.0 (API 21) or later versions.

## VII. Model Downloads (Supported ONNX Models)

| Model Name | Type | Download Link |
| ---- | ---- | ---- |
| silero-vad-v6-onnx | Streaming/Non-streaming | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v6-onnx) |
| silero-vad-v5-onnx | Streaming/Non-streaming | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v5-onnx) |
| silero-vad-onnx | Streaming/Non-streaming | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-onnx) |

## VIII. References
[1] [Silero VAD](https://github.com/snakers4/silero-vad) 