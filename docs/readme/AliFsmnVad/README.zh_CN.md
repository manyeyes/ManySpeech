# AliFsmnVad

AliFsmnVad 是一款用于 **Fsmn-Vad 模型解码**的 C# 库，核心用途是实现**语音活动检测（Voice Activity Detection, VAD）**，精准识别音频中的有效语音片段。


## 1. 简介
AliFsmnVad 基于 C# 开发，通过调用 `Microsoft.ML.OnnxRuntime` 组件实现对 ONNX 格式 Fsmn-Vad 模型的高效解码，具备以下核心特性：
- **兼容性优异**：支持 .NET Framework 4.6.1+、.NET 6.0+ 等框架，可跨 Windows、macOS、Linux、Android、iOS 等平台编译，且支持 AOT 编译，部署灵活。
- **性能高效**：语音端点检测全流程的实时因子（RTF）约为 0.008，处理速度远高于实时音频流需求。
- **功能聚焦**：作为 16kHz 通用 VAD 工具，基于达摩院语音团队提出的「FSMN-Monophone VAD」高效模型，可精准检测长语音片段中有效语音的起止时间点；通过提取有效音频片段输入识别引擎，能显著减少无效语音带来的识别误差，提升语音识别准确性。


## 2. 安装方式
通过 NuGet 包管理器安装（推荐）：

### 2.1 使用 Package Manager Console
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.AliFsmnVad
```

### 2.2 使用 .NET CLI
在命令行中执行：
```bash
dotnet add package ManySpeech.AliFsmnVad
```

### 2.3 手动安装
在 NuGet 包管理器界面搜索「ManySpeech.AliFsmnVad」，点击「安装」即可。


## 3. VAD 常用参数调整说明
参数配置参考项目中的 `vad.yaml` 文件，核心可调整参数如下（需根据实际场景优化）：

### 3.1 max_end_silence_time
- **功能**：尾部连续检测到静音时，触发尾点判停的时间阈值。
- **参数范围**：500ms ～ 6000ms
- **默认值**：800ms
- **注意事项**：值过低易导致有效语音被提前截断，值过高会保留过多静音片段，需根据音频场景（如会议录音、单人讲话）平衡。

### 3.2 speech_noise_thres
- **功能**：判断“语音/噪音”的核心阈值，当「speech 得分 - noise 得分」大于此值时，判定为有效语音。
- **参数范围**：(-1, 1)
- **调节逻辑**：
  - 取值越趋近 -1：噪音被误判为语音的概率越高（FA，False Alarm 误检率上升）；
  - 取值越趋近 +1：有效语音被误判为噪音的概率越高（Pmiss，Miss Probability 漏检率上升）；
- **建议**：通常需根据模型在目标场景长语音测试集上的效果，取“误检率-漏检率”平衡值。


## 4. 调用方式
以下为完整调用流程示例，包含命名空间引用、初始化、核心调用及结果获取。

### 4.1 引入命名空间
安装完成后，在代码文件头部引入命名空间：
```csharp
using ManySpeech.AliFsmnVad;
using ManySpeech.AliFsmnVad.Model;
```

### 4.2 初始化模型和配置
需提前准备 3 个核心文件（模型文件、配置文件、均值归一化文件），初始化时指定文件路径及批量解码参数：
```csharp
// 获取应用程序根目录（避免硬编码路径）
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelDirName = "speech_fsmn_vad_zh-cn-16k-common-onnx"; // 模型文件夹名称

// 拼接模型、配置、均值归一化文件的完整路径
string modelFilePath = Path.Combine(applicationBase, modelDirName, "model.onnx");
string configFilePath = Path.Combine(applicationBase, modelDirName, "vad.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelDirName, "vad.mvn");

int batchSize = 2; // 批量解码大小（根据硬件性能调整，建议 1~4）

// 初始化 AliFsmnVad 实例（加载模型并配置参数）
AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
```

### 4.3 调用核心方法
根据音频文件大小选择不同的调用方式，小文件适合一次性处理，大文件建议分步处理以降低内存占用：

#### 方法 1：小文件一次性处理
适用于音频时长较短（如 < 5 分钟）的场景：
```csharp
// samples：音频采样数据（需提前通过 NAudio 等库读取，格式为 float[]）
SegmentEntity[] segments = aliFsmnVad.GetSegments(samples);
```

#### 方法 2：大文件分步处理
适用于音频时长较长（如 > 5 分钟）的场景，分步读取并处理音频数据：
```csharp
// samples：音频采样数据（float[] 类型，可分批次读取）
SegmentEntity[] segments = aliFsmnVad.GetSegmentsByStep(samples);
```

### 4.4 获取并解析结果
`SegmentEntity` 数组包含每段有效语音的核心信息，可遍历提取语音片段和时间戳：
```csharp
// 遍历所有有效语音片段
foreach (SegmentEntity segment in segments)
{
    // segment.Waveform：当前片段的音频采样数据（float[]，可直接输入语音识别引擎）
    // segment.Segment：当前片段的起止时间戳（毫秒级，格式为 [起始时间, 结束时间]）
    Console.WriteLine($"有效语音片段：{segment.Segment[0]}ms ~ {segment.Segment[1]}ms");
}
```

#### 输出示例
```text
load model and init config elapsed_milliseconds: 463.5390625
vad infer result:
[[70,2340], [2620,6200], [6480,23670], [23950,26250], [26780,28990], [29950,31430], [31750,37600], [38210,46900], [47310,49630], [49910,56460], [56740,59540], [59820,70450]]
elapsed_milliseconds: 662.796875
total_duration: 70470.625ms
rtf: 0.009405292985552491
```
- 时间戳格式：`[起始时间, 结束时间]`（单位：毫秒），例如 `[70,2340]` 表示 70ms ~ 2340ms 为一段有效语音，静音/噪音片段已自动过滤。


## 5. 语音识别衔接
将 `SegmentEntity.Waveform` 作为输入参数，可对接主流语音识别库执行后续识别任务，支持的库包括：
- AliParaformerAsr
- K2TransducerAsr
- SherpaOnnxSharp（调用其 `offlineRecognizer` 相关方法）

具体调用示例可参考对应库的官方文档或 `ManySpeech.AliFsmnVad.Examples` 测试项目。


## 6. 其他说明
### 6.1 测试用例
提供独立测试项目 `ManySpeech.AliFsmnVad.Examples`，包含完整的音频读取、VAD 检测、结果解析示例，可直接参考调试。

### 6.2 支持平台
- Windows：Windows 7 SP1 及以上版本
- macOS：macOS 10.13 (High Sierra) 及以上版本
- Linux：支持 .NET 6.0+ 官方兼容的 Linux 发行版（需提前安装依赖库，详见 [.NET 官方文档](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)）
- Android：Android 5.0 (API 21) 及以上版本
- iOS：需配合 Xamarin 或 .NET MAUI 开发，支持 iOS 11.0 及以上版本

### 6.3 依赖说明
示例中音频采样数据（`samples`）的读取与处理依赖 **NAudio 库**，需通过 NuGet 安装：
```bash
Install-Package NAudio
```


## 7. 模型下载
可从以下平台下载官方 Fsmn-Vad 模型（16kHz 通用版）：
- Hugging Face：[manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx](https://huggingface.co/manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx)
- ModelScope：[manyeyes/alifsmnvad-onnx](https://www.modelscope.cn/models/manyeyes/alifsmnvad-onnx)

官方模型详情可参考：[damo/speech_fsmn_vad_zh-cn-16k-common-onnx](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx)


## 8. 参考
- [FunASR 官方仓库](https://github.com/modelscope/FunASR)（Fsmn-Vad 模型源自此项目）