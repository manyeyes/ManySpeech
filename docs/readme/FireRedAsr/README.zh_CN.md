# FireRedAsr
FireRedAsr是一个用于解码FireRedASR AED-L模型的c#库，用于语音识别（ASR）

##### 简介：
FireRedAsr 是一款基于 C# 开发的语音识别（ASR）库。该库兼容性优异，支持net461+、.NET6.0+等框架版本，不仅专注于高效完成语音识别任务，还具备出色的跨平台能力，支持在 Windows、Linux、Android、macOS、iOS 等多平台上进行编译与调用，适配场景广泛。
其核心依赖的 FireRedASR-AED 模型，以平衡高性能与计算效率为设计目标，采用基于注意力机制的编码器 - 解码器（AED）架构，能够作为基于大语言模型（LLM）的语音模型中的高效语音表示模块，为语音识别任务提供稳定且优质的技术支撑

##### 支持的模型（ONNX）
| 模型名称  |  类型 |  支持语言  | 下载地址  |
| ------------ | ------------ | ------------ | ------------ |
|  fireredasr-aed-large-zh-en-onnx-offline-20250124 | 非流式  | 中文、英文  |[modelscope](https://www.modelscope.cn/models/manyeyes/fireredasr-aed-large-zh-en-onnx-offline-20250124 "modelscope") |

##### 如何使用
###### 1.克隆项目源码
```bash
cd /path/to
git clone https://github.com/manyeyes/FireRedASR.git
```
###### 2.下载上述列表中的模型到目录：/path/to/FireRedASR/FireRedASR.Examples
```bash
cd /path/to/FireRedASR/FireRedASR.Examples
git clone https://www.modelscope.cn/manyeyes/[模型名称].git
```
###### 3.使用vs2022(或其他IDE)加载工程，
###### 4.将模型目录中的文件设置为：复制到输出目录->如果较新则复制
###### 5.修改示例中代码：string modelName =[模型目录名]
###### 6.运行项目
###### 7.如何调用
参考:FireRedAsrExamples.cs中的示例代码
###### 8.运行结果
```
朱立南在上市见面会上表示

这是第一种第二种叫呃与always always什么意思啊

好首先说一下刚才这个经理说完了这个销售问题咱再说一下咱们的商场问题首先咱们商场上半年业这个先各部门儿汇报一下就是业绩

elapsed_milliseconds:4391.234375
total_duration:21015.0625
rtf:0.2089565222563578
Hello, World!
```
###### 相关工程：
* 语音端点检测，解决长音频合理切分的问题，项目地址：[AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* 文本标点预测，解决识别结果没有标点的问题，项目地址：[AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

###### 其他说明：

测试用例：FireRedASR.Examples。
测试CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
支持平台：
Windows 7 SP1或更高版本,
macOS 10.13 (High Sierra) 或更高版本,ios等，
Linux 发行版（需要特定的依赖关系，详见.NET 6支持的Linux发行版列表），
Android（Android 5.0 (API 21) 或更高版本）。

引用参考
----------
[1] https://github.com/FireRedTeam/FireRedASR
