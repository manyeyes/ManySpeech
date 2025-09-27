 ( 简体中文 | [English](README.EN.md))

# SileroVad

C# 语音端点检测库，用于基于 Silero VAD 模型的语音活动检测。

## 简介

**SileroVad** 是一个采用 C# 开发的语音端点检测（VAD）库，底层基于 `Microsoft.ML.OnnxRuntime` 实现 ONNX 模型解码。该库具有以下特点：
- 多框架支持：兼容 .NET Framework 4.6.1+、.NET 6.0+、.NET Core 3.1 及 .NET Standard 2.0+
- 跨平台能力：支持 Windows、macOS、Linux 等系统，可进行跨平台编译
- 部署灵活：支持 AOT（Ahead-of-Time）编译，使用简单便捷


## 支持的模型（ONNX）

| 模型名称              | 类型         | 下载地址                                                                 |
|-----------------------|--------------|--------------------------------------------------------------------------|
| silero-vad-v6-onnx    | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v6-onnx)   |
| silero-vad-v5-onnx    | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-v5-onnx)   |
| silero-vad-onnx       | 流式/非流式  | [ModelScope](https://modelscope.cn/models/manyeyes/silero-vad-onnx)      |


## 快速开始

### 1. 克隆项目源码
```bash
cd /path/to/your/workspace
git clone https://github.com/manyeyes/SileroVad.git
```

### 2. 下载模型文件
将上述表格中的模型下载至示例项目目录：
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
# 替换 [模型名称] 为实际模型名（如 silero-vad-v6-onnx）
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```

### 3. 配置项目
- 使用 Visual Studio 2022 或其他兼容 IDE 加载解决方案
- 将模型目录中的所有文件设置为：**复制到输出目录 -> 如果较新则复制**

### 4. 准备语音识别模型（可选）
示例中通过 `OfflineRecognizer` 方法实现语音识别，需额外下载 ASR 模型：
```bash
cd /path/to/your/workspace/SileroVad/SileroVad.Examples
git clone https://www.modelscope.cn/manyeyes/aliparaformerasr-large-zh-en-timestamp-onnx-offline.git
```

### 5. 运行示例
- 修改示例代码中的模型路径：`string modelName = "[模型目录名]"`
- 程序入口：`Program.cs`，默认执行两个测试用例：
  - 非流式检测：`TestOfflineVad()`（源码：`OfflineVad.cs`）
  - 流式检测：`TestOnlineVad()`（源码：`OnlineVad.cs`）


## 运行效果

### 非流式检测输出
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

### 流式检测输出
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


## 相关项目

- **语音识别**：[AliParaformerAsr](https://github.com/manyeyes/AliParaformerAsr)
- **语音端点检测（长音频切分）**：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad)
- **文本标点预测**：[AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc)


## 系统要求

- **测试环境**：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz 2.59 GHz
- **支持平台**：
  - Windows：Windows 7 SP1 及以上版本
  - macOS：macOS 10.13 (High Sierra) 及以上版本（含 iOS）
  - Linux：兼容 .NET 6 支持的发行版（需满足特定依赖）
  - Android：Android 5.0 (API 21) 及以上版本


## 引用参考
[1] [Silero VAD](https://github.com/snakers4/silero-vad)