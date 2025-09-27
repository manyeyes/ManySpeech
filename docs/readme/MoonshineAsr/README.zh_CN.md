# MoonshineAsr
c# library for decoding moonshine's tiny, base Models，used in speech recognition (ASR)

##### 简介：

MoonshineAsr是一个使用C#编写的“语音识别”库，底层调用Microsoft.ML.OnnxRuntime对onnx模型进行解码，支持 net461+、net60+、netcoreapp3.1 及 netstandard2.0+ 等多种环境，支持跨平台编译，支持AOT编译。使用简单方便。

##### 支持的模型（ONNX）

| 模型名称  |  类型 | 支持语言  | 标点  |  时间戳 | 下载地址  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  moonshine-base-en-onnx | 非流式  | 英文  |  是 | 否  |  [modelscope](https://modelscope.cn/models/manyeyes/moonshine-base-en-onnx "modelscope") |
|  moonshine-tiny-en-onnx | 非流式  | 英文  |  是 | 否  | [modelscope](https://modelscope.cn/models/manyeyes/moonshine-tiny-en-onnx "modelscope") |

##### 如何使用
###### 1.克隆项目源码
```bash
cd /path/to
git clone https://github.com/manyeyes/MoonshineAsr.git
```
###### 2.下载上述列表中的模型到目录：/path/to/MoonshineAsr/MoonshineAsr.Examples
```bash
cd /path/to/MoonshineAsr/MoonshineAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```
###### 3.使用vs2022（或其IDE）加载项目，并运行MoonshineAsr.Examples
```bash
// 三种使用方式
// 1.直接一次识别单个音频文件（建议文件小一点识别更快）
test_MoonshineAsrOfflineRecognizer();
// 2.分片输入识别，适用于外接vad
test_MoonshineAsrOnlineRecognizer();
// 3.流式输入识别，使用内置vad功能，自动断句，更加便捷
test_MoonshineAsrOnlineVadRecognizer();
```

```bash
// 下载vad模型
cd /path/to/MoonshineAsr/MoonshineAsr.Examples
git clone https://www.modelscope.cn/manyeyes/alifsmnvad-onnx.git
```
###### 使用流式输入的方式识别，识别结果（自带时间戳）：
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

 引用参考
----------
[1] https://github.com/usefulsensors/moonshine
