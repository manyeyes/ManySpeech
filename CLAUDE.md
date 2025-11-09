# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ManySpeech 是一个基于 C# 的跨平台语音处理套件，使用 ONNX 模型提供语音识别、语音端点检测、标点恢复、音频分离和增强功能。项目支持多种 .NET 框架，从 .NET 4.6.1 到 .NET 9.0，包括移动平台支持。

## 常用开发命令

### 编译项目
```bash
# 停止可能运行的进程（重要：根据用户要求，编译前必须先 kill 进程）
pkill -f "ManySpeech" || pkill -f "dotnet"

# 编译整个解决方案
dotnet build ManySpeech.sln

# 编译特定项目
dotnet build src/ManySpeech.AliParaformerAsr/ManySpeech.AliParaformerAsr.csproj

# 清理并重新编译
dotnet clean ManySpeech.sln
dotnet build ManySpeech.sln
```

### 运行示例
```bash
# CLI 示例（识别器）
dotnet run --project samples/ManySpeech.Cli.Sample/ManySpeech.Cli.Sample.csproj

# MAUI 示例（需要特定环境）
dotnet build samples/ManySpeech.Maui.Sample/ManySpeech.Maui.Sample.csproj
```

### 测试
```bash
# 运行所有测试
dotnet test

# 运行特定测试项目
dotnet test [测试项目路径]
```

## 项目架构

### 核心模块结构

1. **语音识别模块** (src/ManySpeech.*Asr/)
   - `ManySpeech.AliParaformerAsr`: Paraformer 和 SenseVoice 模型支持
   - `ManySpeech.WhisperAsr`: Whisper 系列模型
   - `ManySpeech.FireRedAsr`: FireRedASR 模型
   - `ManySpeech.K2TransducerAsr`: Kaldi Zipformer 模型
   - `ManySpeech.WenetAsr`: WeNet 模型
   - `ManySpeech.MoonshineAsr`: Moonshine 模型

2. **语音处理模块** (src/ManySpeech.*Vad/, src/ManySpeech.*Punc/, src/ManySpeech.AudioSep/)
   - `ManySpeech.AliFsmnVad`: FSMN-VAD 端点检测
   - `ManySpeech.SileroVad`: Silero VAD 端点检测
   - `ManySpeech.AliCTTransformerPunc`: CT-Transformer 标点恢复
   - `ManySpeech.AudioSep`: 音频分离和增强

3. **工具模块** (src/)
   - `AudioInOut`: 音频输入输出处理
   - `PreProcessUtils`: 预处理工具
   - `ManySpeech.SpeechProcessing`: 语音处理核心功能

### 解决方案文件夹结构

- **Core**: 核心语音处理组件
- **Utils**: 工具类库
- **Examples**: 各组件的使用示例
- **Samples**: 完整的示例应用程序
- **Tests**: 测试项目

### 关键设计特点

1. **跨平台支持**: 所有项目支持多目标框架，包括桌面、移动和服务器平台
2. **ONNX 运行时**: 基于 Microsoft.ML.OnnxRuntime 进行模型推理
3. **模块化设计**: 每个语音处理功能都是独立的 NuGet 包
4. **统一接口**: 相似功能的组件遵循一致的接口设计

### 模型管理

- 模型通过 GitHelper 类自动下载和管理
- 默认模型存储在应用程序目录或通过 `-base` 参数指定的目录
- 支持量化模型 (int8) 和全精度模型 (fp32)

### 配置文件

项目目前没有使用 appsettings.json 配置文件。配置通过以下方式管理：
1. 命令行参数
2. 环境变量 (前缀: `MANYSPEECH_`)
3. 代码中的默认配置

## 重要注意事项

1. **编译前必须停止进程**: 按照用户要求，执行 dotnet build 或 run 操作前必须先 kill 掉相关进程
2. **仅编译不运行**: 用户明确要求只编译通过，不要执行 dotnet run，用户会自己执行
3. **多配置文件**: 如果需要修改配置，需要同时修改 appsettings.json 和 appsettings.Development.json（虽然当前项目没有这些文件）
4. **模型依赖**: 运行时需要下载相应的 ONNX 模型文件
5. **语言支持**: 项目支持中文和英文，包括用户界面和语音识别