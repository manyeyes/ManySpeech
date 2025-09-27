（简体中文 | [English](README.md) ）

 <div style="text-align:center;"><img width="128" height="128" align="center" alt="dotnet_bot_128x128" src="https://github.com/user-attachments/assets/d8119208-17a5-4bff-b222-003093ad7d18" /></div>

# ManySpeech 语音处理套件：跨平台 C# 语音解决方案 🎤

## 项目概述 📋

ManySpeech（ https://github.com/manyeyes/ManySpeech ）是由 manyeyes 社区开发的一款基于 C# 的语音处理套件。该项目以优秀的开源模型为核心，依托 Microsoft.ML.OnnxRuntime 实现 ONNX 模型解码，旨在解决三大关键问题：
- 🚩 跨平台部署的兼容性问题
- 🚩 不同场景（实时/离线、多语言）下的模型适配难题
- 🚩 复杂工具链的集成门槛

作为一套平衡“易用性、功能性与部署灵活性”的解决方案，ManySpeech 能有效提升开发效率，为 .NET 生态下的语音处理需求提供强有力的支持。

## 核心特性 🌟

### 1. 贴合 C# 开发者实际需求 👨💻

从 C# 开发的实际场景出发，ManySpeech 在模型覆盖、平台兼容、开发流程等维度高度契合 .NET 生态，可作为工具选型的重要参考。

### 2. 多场景模型覆盖 🧩

ManySpeech 通过 **“语音识别、端点检测、标点恢复、音频分离增强”** 等多任务协同，无需整合多套工具链——仅靠组件组合即可匹配不同业务需求，有效解决“场景多变导致工具碎片化”的问题。

### 3. 全平台兼容，降低多端部署成本 🌐

尽管 C# 依托 .NET 生态已具备成熟的跨平台能力，但语音处理因深度依赖底层资源（如 ONNX 运行时、音频接口），仍面临诸多挑战。ManySpeech 针对性缓解了这些跨平台痛点：
- **广泛的框架支持** 🛠️：兼容 .NET 4.6.1+、.NET 6.0+、.NET Core 3.1、.NET Standard 2.0+，覆盖从传统桌面开发到跨平台开发的主流框架。
- **全面的系统适配** 💻📱：可运行于 Windows 7+、macOS 10.13+、Linux 各发行版（符合 .NET 6 支持列表）、Android 5.0+、iOS，无论是桌面端软件、服务器批量处理，还是移动端 APP，均能稳定部署。
- **轻量化优化** ⚡：支持 AOT 编译，编译后可执行文件体积减少 30%+，启动速度提升约 20%，适配嵌入式设备（如物联网语音控制终端）及轻量化部署场景。

### 4. 适配多任务，组件规范统一，协同灵活自由 🤝

#### （1）语音识别任务 🎙️→📝
**核心功能**：实现语音到文本的转录。
- 流式模型：延迟低，适合实时交互（在线客服、语音输入法） ⚡
- 非流式模型：适合离线转写（本地录音、视频字幕） 📹
- 多语言支持：覆盖中文、英文，额外支持粤语、日语、韩语（如 SenseVoice 模型） 🌍
- 精准特性：支持字级（中文）/单词级（英文）时间戳（精度至毫秒），且支持自定义热词（如 paraformer-seaco-large-zh-timestamp-onnx-offline），可快速适配“金融术语”“医疗名词”等垂直领域场景 🔍
- 智能标点：SenseVoiceSmall 与 whisper-large 模型支持多语言识别，且自带标点预测，省去“识别后手动加标点”的操作 ✏️

**相关组件**：
- ManySpeech.AliParaformerAsr：支持 Paraformer、SenseVoice 等 ONNX 模型
- ManySpeech.FireRedAsr：支持 FireRedASR-AED-L 等 ONNX 模型
- ManySpeech.K2TransducerAsr：支持新一代 Kaldi 中的 Zipformer 等 ONNX 模型
- ManySpeech.MoonshineAsr：支持 Moonshine 中的 Tiny、Base 等 ONNX 模型
- ManySpeech.WenetAsr：支持 WeNet 中的 ONNX 模型
- ManySpeech.WhisperAsr：支持 whisper 系列 ONNX 模型，具备语音语言识别能力

#### （2）语音端点检测任务 🎯
**核心功能**：精准检测长语音片段中有效语音的起止时间点。
- 通过提取有效音频片段并输入识别引擎，可显著减少无效语音带来的识别误差，提升语音识别任务的准确性 📊
- 适用场景：长语音离线转写（如会议录音处理）、实时语音交互（如智能客服实时响应） 🗣️

**相关组件**：
- ManySpeech.AliFsmnVad：支持 Fsmn-Vad 模型，专注于精准检测有效语音的起止时间点
- ManySpeech.SileroVad：支持 Silero-VAD 模型，核心功能为精准检测长语音片段中有效语音的起止时间点

#### （3）标点恢复任务 ✏️
**核心功能**：为文本（尤其是语音识别模型输出的无标点文本）自动预测并添加标点符号。
- 通过后处理优化文本结构，提升内容的可读性与连贯性 📄

**相关组件**：
- ManySpeech.AliCTTransformerPunc：支持 CT-Transformer 模型，可用于语音识别模型输出文本的标点预测

#### （4）声源分离、语音增强任务 🎧
**核心功能**：专注于音频分离、降噪与增强。
- 能从混合音频中分离出目标声音（如人声与背景音分离），降低环境噪音干扰 🚫🔊
- 提升目标语音的清晰度，为后续语音识别、音频处理等任务提供高质量音频输入 📈
- 适用场景：会议录音去杂音、嘈杂场景下的语音提取、多说话人音频分离 🎤👥

**相关组件**：
- ManySpeech.AudioSep：支持 clearervoice、gtcrn、spleeter、uvr 中的 ONNX 模型，专注于实现音频分离、降噪与增强功能

## 快速上手 🚀

查看我们的 **“示例程序”**，了解如何使用 ManySpeech 开发应用程序。

## 支持者 🙏

感谢所有支持我们的人！您的支持是 ManySpeech 持续完善的动力。

## ManySpeech —— 使用 C# 开发人工智能语音应用 🤖

ManySpeech 将持续集成前沿 AI 模型，为开发者提供低门槛的企业级语音处理集成路径。开发者可根据项目具体需求（如是否需要实时响应、是否涉及方言），结合不同模型的特性，选择最适配的方案，高效推进功能落地。
