 ( 简体中文 | [English](README.md) )

# ManySpeech.WenetAsr 使用指南

## 一、简介
ManySpeech.WenetAsr 是一个用于解码 Wenet ASR ONNX 模型的 C# 库，具备以下特点：
- **环境兼容性**：支持 net461+、net60+、netcoreapp3.1 及 netstandard2.0+ 等多种环境。
- **跨平台特性**：支持跨平台编译，可在 Windows、macOS、Linux、Android、iOS 等系统运行。
- **AOT 支持**：支持 AOT 编译，使用简单便捷。
- **多平台应用构建**：可结合 MAUI 或 Uno 快速构建多平台应用程序。

## 二、安装方式
### （一）通过 NuGet 包管理器安装（推荐）
1. **使用 Package Manager Console**
在 Visual Studio 的「Package Manager Console」中执行以下命令：
```bash
Install-Package ManySpeech.WenetAsr
```
2. **使用.NET CLI**
在命令行中执行：
```bash
dotnet add package ManySpeech.WenetAsr
```
3. **手动安装**
在 NuGet 包管理器界面搜索「ManySpeech.WenetAsr」，点击「安装」即可。

### （二）通过 XML 节点手动添加引用（适用于手动管理项目配置场景）
若需通过直接修改项目配置文件（如 `.csproj` 文件）的 XML 节点添加引用，操作如下：
1. 找到项目根目录下的 `.csproj` 文件（如 `YourProject.csproj`），用文本编辑器或 IDE 打开。
2. 在 `<Project>` 根节点下的 `<ItemGroup>` 节点中，添加以下 XML 引用节点（若项目中无 `<ItemGroup>` 节点，可手动创建）：
```xml
<ItemGroup>
  <!-- 添加 ManySpeech.WenetAsr 引用，Version 需替换为最新版本号，可从 NuGet 官网查询 -->
  <PackageReference Include="ManySpeech.WenetAsr" Version="x.x.x" />
</ItemGroup>
```
3. 保存 `.csproj` 文件后，重启 IDE 或在命令行执行 `dotnet restore` 命令，触发依赖包还原，完成引用添加。

> 说明：`Version` 属性需填写实际需要的版本号（如 `1.0.0`，可通过 [NuGet 官网](https://www.nuget.org/packages/ManySpeech.WenetAsr) 查询最新版本），确保版本与项目的 .NET 环境兼容。


## 三、WenetAsr 与 WenetAsr2 的区别
### （一）共同之处
- 功能相同，均支持流式（streaming）和非流式（non-streaming）模型解码。
- 调用方式一致。

### （二）不同之处
| 库名称 | 流式与非流式模型加载模块 | 模型与扩展 |
| ---- | ---- | ---- |
| ManySpeech.WenetAsr | 合二为一 | 1. 加载 Wenet 官方导出的 ONNX 模型<br>2. 代码简洁 |
| ManySpeech.WenetAsr2 | 各自独立 | 1. 加载 Wenet 官方导出的 ONNX 模型<br>2. 便于扩展，若自定义导出的流式和非流式模型参数不同，可在各自模块调整而互不影响 |

**推荐使用**：如果没有二次开发需求，直接使用 Wenet 官方导出的 ONNX 模型，建议选择 WenetAsr。

## 四、代码调用方法
### （一）离线（非流式）模型调用方法
1. **添加项目引用**
```csharp
using ManySpeech.WenetAsr;
using ManySpeech.WenetAsr.Model;
```
2. **模型初始化和配置**
```csharp
// 加载模型
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
```
3. **调用过程**
```csharp
// 此处省略音频文件到 sample 的转换，具体可参考 examples 中的 test_WenetAsrOfflineRecognizer
OfflineStream stream = offlineRecognizer.CreateOfflineStream();
stream.AddSamples(sample);
Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
Console.WriteLine(result.Text);
```
4. **输出结果示例**
- **中文模型** (wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506)
```
正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

啊这是第一种第二种叫嗯与欧维斯欧维斯什么意思

蒋永伯被拍到带着女儿出游

周望君就落实控物价

每当新年的钟声敲响的时候我总会闭起眼睛静静地许愿有时也会给自己定下新年的奋斗目标还有时听到新年的终身时我的心里会有一种遗憾的感觉感慨时光过的如此匆匆而自己往年的愿 望还没达成尽管如此经过岁月的洗礼我已长大成熟学会了勇敢的面对现实的一切

elapsed_milliseconds:6729.828125
total_duration:57944.0625
rtf:0.11614353282530027
```
- **英文模型** (wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728)
```
after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels

god as a direct consequence of the sin which man thus punished had given her a lovely child whose place was on that same dishonored bosom to connect her parent for ever with the race and descent of mortals and to be finally a blessed soul in heaven

elapsed_milliseconds:2639.1171875
total_duration:23340
rtf:0.11307271583119109
```

### （二）实时（流式）模型调用方法
1. **添加项目引用**
```csharp
using ManySpeech.WenetAsr;
using ManySpeech.WenetAsr.Model;
```
2. **模型初始化和配置**
```csharp
// 加载模型
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "wenet-u2pp-conformer-wenetspeech-onnx-online-20220506";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.int8.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
```
3. **调用过程**
```csharp
// 此处省略音频文件到 sample 的转换，或处理来自麦克风的音频
// 具体实现可参考 ManySpeech.WenetAsr.Examples 中的 test_WenetAsrOnlineRecognizer 示例代码
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
while (true)
{
    // 这是一个简单的解码示意，如需了解更详细流程，请参考 examples
    // sample = 来自音频文件或麦克风的音频数据
    stream.AddSamples(sample);
    OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
    Console.WriteLine(result.Text);
}
```
4. **输出结果示例**
```



正是因为

正是因为存在绝

正是因为存在绝对正义

正是因为存在绝对正义所以我们

正是因为存在绝对正义所以我们接受现

正是因为存在绝对正义所以我们接受现实的相对

正是因为存在绝对正义所以我们接受现实的相对正议

正是因为存在绝对正义所以我们接受现实的相对正议但

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界

正是因为存在绝对正义所以我们接受现实的相对正议但是不要因为现实的相对正义我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

elapsed_milliseconds:2148.8515625
total_duration:13052
rtf:0.16463772314587802
```

## 五、相关工程
- **语音端点检测**：为解决长音频合理切分问题，可添加 ManySpeech.AliFsmnVad 库，安装命令：
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **文本标点预测**：针对识别结果缺乏标点的情况，可添加 ManySpeech.AliCTTransformerPunc 库，安装命令：
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
具体调用示例可参考对应库的官方文档或 `ManySpeech.WenetAsr.Examples` 项目，该项目是控制台/桌面端示例，展示了离线转写、实时识别等基础功能。

## 六、其他说明
- **测试用例**：以 `ManySpeech.WenetAsr.Examples` 作为测试用例。
- **测试 CPU**：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz（2.59 GHz）。
- **支持平台**：
  - **Windows**：Windows 7 SP1 及更高版本。
  - **macOS**：macOS 10.13 (High Sierra) 及更高版本，支持 iOS。
  - **Linux**：适用于 Linux 发行版（需满足特定依赖关系，详见.NET 6 支持的 Linux 发行版列表）。
  - **Android**：Android 5.0 (API 21) 及更高版本。

## 七、模型下载（支持的 ONNX 模型）
| 模型名称 | 类型 | 支持语言 | 下载地址 |
| ---- | ---- | ---- | ---- |
| wenet-u2pp-conformer-aishell-onnx-online-20210601 | 流式 | 中文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-online-20210601 "modelscope") |
| wenet-u2pp-conformer-aishell-onnx-offline-20210601 | 离线 | 中文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-aishell-onnx-offline-20210601 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 | 流式 | 中文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-online-20220506 "modelscope") |
| wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 | 离线 | 中文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 | 流式 | 英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-online-20210728 "modelscope") |
| wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 | 离线 | 英文 | [modelscope](https://www.modelscope.cn/models/manyeyes/wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728 "modelscope") |

**引用参考**：
[1] https://github.com/wenet-e2e/wenet