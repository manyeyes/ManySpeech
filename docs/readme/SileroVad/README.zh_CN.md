 ( 简体中文 | [English](README.md) )

# ManySpeech.SileroVad 使用指南

## 一、简介
**ManySpeech.SileroVad** 是一个采用 C# 开发的语音端点检测（VAD）库，底层基于 `Microsoft.ML.OnnxRuntime` 实现 ONNX 模型解码。该库具备以下特点：
- **多环境支持**：可兼容 net461+、net60+、netcoreapp3.1 以及 netstandard2.0+ 等多种环境，能适配不同开发场景的需求。
- **跨平台编译特性**：支持跨平台编译，无论是 Windows、macOS 还是 Linux、Android 等系统，都能进行编译使用，拓展了应用的范围。
- **支持 AOT 编译**：使用起来简单便捷，方便开发者快速集成到项目中。

## 二、安装方式
推荐通过 NuGet 包管理器进行安装，以下为三种具体安装途径：

### （一）使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.SileroVad
```

### （二）使用.NET CLI
在命令行中输入以下命令来安装：
```bash
dotnet add package ManySpeech.SileroVad
```

### （三）手动安装
在 NuGet 包管理器界面搜索「ManySpeech.SileroVad」，点击「安装」即可。

## 三、快速开始

### （一）下载 VAD 模型
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
# 替换 [模型名称] 为实际模型名（如 silero-vad-v6-onnx）
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```

### （二）下载 ASR 模型（可选）
示例中通过 `OfflineRecognizer` 方法实现语音识别，需额外下载 ASR 模型：
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
git clone https://www.modelscope.cn/manyeyes/aliparaformerasr-large-zh-en-timestamp-onnx-offline.git
```

### （三）配置项目
1. **使用 Visual Studio 2022 或其他兼容 IDE 加载解决方案**：利用合适的开发工具打开项目解决方案，为后续操作做好准备。
2. **将模型目录中的所有文件设置为：复制到输出目录 -> 如果较新则复制**：确保模型文件能正确地随项目输出，以便在运行时可正常调用，保证项目能顺利获取到所需的模型资源。

### （四）引入命名空间
安装完成后，在代码文件头部引入以下命名空间：
```csharp
using ManySpeech.SileroVad;
using ManySpeech.SileroVad.Model;
```

### （五）初始化模型和配置
需提前准备 3 个核心文件（模型文件、配置文件、均值归一化文件），初始化时指定文件路径及批量解码参数：
```csharp
// 获取应用程序根目录（避免硬编码路径）
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "silero-vad-v6-onnx"; // 模型文件夹名称

// 拼接模型、配置、均值归一化文件的完整路径
string modelFilePath = applicationBase + "./" + modelName + "/silero_vad.onnx";
string configFilePath = applicationBase + "./" + modelName + "/vad.yaml";

int batchSize = 2; // 批量解码大小（根据硬件性能调整，建议 1~4）

// 初始化 AliFsmnVad 实例（加载模型并配置参数）
AliFsmnVad aliFsmnVad = new OfflineVad(modelFilePath, configFilePath: configFilePath, threshold: 0F, isDebug: false);
```

### （六）调用核心方法
根据音频文件大小选择不同的调用方式，小文件适合一次性处理，大文件建议分步处理以降低内存占用：
```csharp
// samples：音频采样数据（需提前通过 NAudio 等库读取，格式为 float[]）
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
        //使用 ManySpeech.AliParaformerAsr 库对切分后的 samples 进行识别
        //OfflineRecognizer(new List<float[]>() { item.Second });
        Console.WriteLine("");
    }

}
```

### （七）直接运行示例看效果
1. **修改示例代码中的模型路径**：将 `string modelName = "[模型目录名]"` 修改为实际对应的模型目录名称，确保程序能准确找到并加载模型。
2. **程序入口**：程序入口为 `Program.cs`，默认执行两个测试用例：
    - **非流式检测**：`TestOfflineVad()`（源码：`OfflineVad.cs`）。
    - **流式检测**：`TestOnlineVad()`（源码：`OnlineVad.cs`）。

## 四、运行效果

### （一）非流式检测输出
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

### （二）流式检测输出
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

## 五、相关工程
- **语音识别**：为验证 vad 效果，可添加 ManySpeech.AliParaformerAsr 库对切分后的 samples 进行识别，通过以下命令安装：
```bash
dotnet add package ManySpeech.AliParaformerAsr
```

## 六、系统要求

### （一）测试环境
Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz。

### （二）支持平台
- **Windows**：Windows 7 SP1 及以上版本。
- **macOS**：macOS 10.13 (High Sierra) 及以上版本（含 iOS）。
- **Linux**：兼容.NET 6 支持的发行版（需满足特定依赖）。
- **Android**：Android 5.0 (API 21) 及以上版本。

## 七、模型下载（支持的 ONNX 模型）

| 模型名称              | 类型         | 下载地址                                                                 |
|-----------------------|--------------|--------------------------------------------------------------------------|
| silero-vad-v6-onnx    | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v6-onnx)   |
| silero-vad-v5-onnx    | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v5-onnx)   |
| silero-vad-onnx       | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-onnx)      |

## 八、引用参考
[1] [Silero VAD](https://github.com/snakers4/silero-vad) 