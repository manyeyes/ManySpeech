# MoonshineAsr

MoonshineAsr 是一款基于 C# 开发的**语音识别（ASR）库**，核心用于解码 Moonshine 系列的 tiny/base 版本 ONNX 模型，为语音转文字场景提供高效解决方案。


## 1. 简介
MoonshineAsr 底层依赖 `Microsoft.ML.OnnxRuntime` 实现 ONNX 模型的解码，具备“高兼容性”“易部署”“好上手”三大核心优势，具体特点如下：
- **框架支持广泛**：兼容 .NET Framework 4.6.1+、.NET 6.0+、.NET Core 3.1 及 .NET Standard 2.0+，覆盖绝大多数 C# 项目环境；
- **跨平台与 AOT 友好**：支持 Windows、macOS、Linux 跨平台编译，同时支持 AOT 编译（提前编译为机器码），适配桌面、服务器等多种部署场景；
- **使用门槛低**：提供直观的 API 接口和完整的示例项目，无需深入理解模型细节，即可快速集成语音识别功能。


## 2. 支持的 ONNX 模型
该库仅支持 Moonshine 系列的**非流式 ASR 模型**，模型详情如下表所示（均支持自动标点，暂不支持单词级时间戳）：

| 模型名称                | 模型类型   | 支持语言 | 自动标点 | 单词时间戳 | 下载地址                                                                 |
|-------------------------|------------|----------|----------|------------|--------------------------------------------------------------------------|
| moonshine-base-en-onnx  | 非流式     | 英文     | 支持     | 不支持     | [ModelScope](https://modelscope.cn/models/manyeyes/moonshine-base-en-onnx) |
| moonshine-tiny-en-onnx  | 非流式     | 英文     | 支持     | 不支持     | [ModelScope](https://modelscope.cn/models/manyeyes/moonshine-tiny-en-onnx) |

> 说明：若需要“带时间戳的识别结果”，需使用“内置 VAD 的流式识别模式”（见 3.5 节），通过 VAD 划分语音片段并生成片段级时间戳。


## 3. 使用指南
### 3.1 前置准备
在开始前，需确保环境满足以下条件：
- **开发工具**：Visual Studio 2022（或 Rider、VS Code 等支持 C# 的 IDE）；
- **版本控制工具**：Git（用于克隆项目源码和模型仓库）；
- **.NET 环境**：安装对应框架的 SDK（如 .NET 6.0 SDK，可从 [微软官网](https://dotnet.microsoft.com/zh-cn/download) 下载）。


### 3.2 步骤 1：克隆项目源码
打开终端，切换到你的工作目录，执行以下命令克隆 MoonshineAsr 项目：
```bash
# 切换到目标工作目录（示例路径，可自行替换）
cd /path/to/your/workspace

# 克隆项目仓库
git clone https://github.com/manyeyes/MoonshineAsr.git
```


### 3.3 步骤 2：下载 ASR 模型
将 2 节表格中的目标模型下载到示例项目目录（`MoonshineAsr.Examples`），操作命令如下：
```bash
# 切换到示例项目目录
cd /path/to/your/workspace/MoonshineAsr/MoonshineAsr.Examples

# 替换 [MODEL_NAME] 为实际模型名（如 moonshine-tiny-en-onnx）
git clone https://www.modelscope.cn/manyeyes/[MODEL_NAME].git
```

示例（下载 tiny 版英文模型）：
```bash
git clone https://www.modelscope.cn/manyeyes/moonshine-tiny-en-onnx.git
```


### 3.4 步骤 3：（可选）下载 VAD 模型（用于流式识别）
若需要使用“带内置 VAD 的流式识别”（自动断句+生成时间戳），需额外下载 AliFsmnVAD 模型（语音活动检测模型），同样放入示例项目目录：
```bash
# 保持在 MoonshineAsr.Examples 目录下
git clone https://www.modelscope.cn/manyeyes/alifsmnvad-onnx.git
```


### 3.5 步骤 4：加载项目并运行示例
1. 打开 Visual Studio 2022，通过“文件 → 打开 → 项目/解决方案”，选择项目根目录下的 `MoonshineAsr.sln`；
2. 在解决方案资源管理器中，右键选中 `MoonshineAsr.Examples`，选择“设为启动项目”；
3. 运行项目（按 F5），示例中包含三种核心识别模式，可通过调用对应方法使用：

#### 模式 1：离线识别（单个小文件）
适用于**短音频文件**（如 1 分钟内），一次性输入完整音频数据，识别速度更快：
```csharp
// 调用离线识别方法（处理单个小音频文件）
test_MoonshineAsrOfflineRecognizer();
```

#### 模式 2：分片输入识别（适配外部 VAD）
适用于**自定义 VAD 逻辑**的场景，将音频按外部 VAD 划分的片段逐块输入识别：
```csharp
// 调用分片识别方法（需配合外部 VAD 生成音频片段）
test_MoonshineAsrOnlineRecognizer();
```

#### 模式 3：流式识别（内置 VAD，推荐）
最适合**实时语音场景**，内置 VAD 自动过滤静音、划分语音片段，并生成片段级时间戳，使用最便捷：
```csharp
// 调用内置 VAD 的流式识别方法（自动断句+时间戳）
test_MoonshineAsrOnlineVadRecognizer();
```


## 4. 流式识别示例结果（内置 VAD）
内置 VAD 的流式识别会输出**带时间戳的识别结果**，时间戳格式为 `HH:MM:SS,fff`（时:分:秒,毫秒），静音片段已自动过滤，示例如下：
```
[00:00:00,630-->00:00:06,790]
  thank you. Thank you.

[00:00:07,300-->00:00:10,760]
 Thank you everybody. All right, everybody go ahead and have a seat.

[00:00:11,450-->00:00:15,820]
 How's everybody doing today?

[00:00:17,060-->00:00:20,780]
 How about Tim Spicer?

[00:00:24,270-->00:00:30,450]
  I am here with students at Wakefield High School in Arlington, Virginia.

[00:00:31,070-->00:00:40,430]
 And we've got students tuning in from all across America from kindergarten through 12th grade. And I am just so glad

[00:00:40,960-->00:00:48,430]
 that all could join us today and I want to thank Wakefield for being such an outstanding host give yourselves a big round of applause
```


## 5. 参考资料
[1] Moonshine 官方仓库：[https://github.com/usefulsensors/moonshine](https://github.com/usefulsensors/moonshine)