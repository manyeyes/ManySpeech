# ManySpeech.AliParaformerAsr User Guide

## I. Introduction
ManySpeech.AliParaformerAsr is a "speech recognition" library written in C#. It decodes ONNX models by calling Microsoft.ML.OnnxRuntime at the bottom layer. It has several notable features:
- **Multi-environment Support**: It is compatible with multiple environments such as net461+, net60+, netcoreapp3.1, and netstandard2.0+, meeting the needs of different development scenarios.
- **Cross-platform Compilation**: It supports cross-platform compilation, enabling it to be compiled and used on various operating systems like Windows, macOS, Linux, and Android, thus expanding its application range.
- **AOT Compilation Support**: It is simple and convenient to use, facilitating developers to quickly integrate it into their projects.

## II. Installation Methods
It is recommended to install via the NuGet package manager. There are two specific installation approaches as follows:

### (A) Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.AliParaformerAsr
```

### (B) Using.NET CLI
Enter the following command in the command line to install:
```bash
dotnet add package ManySpeech.AliParaformerAsr
```

## III. Configuration Instructions (Refer to the asr.yaml File)
In the asr.yaml configuration file used for decoding, most parameters do not need to be modified. However, there is a specific modifiable parameter:
- `use_itn: true`: When using the SenseVoiceSmall model configuration, enabling this parameter can achieve inverse text normalization. For example, it can convert text like "123" into "one hundred and twenty-three", making the recognized text more in line with the normal reading habits.

## IV. Code Calling Methods

### (A) Offline (Non-streaming) Model Calling
1. **Adding Project References**
Add the following references in the code:
```csharp
using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
```
2. **Model Initialization and Configuration**
    - **Initialization Method for the paraformer Model**:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model_quant.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
    - **Initialization Method for the SeACo-paraformer Model**:
        - First, find the hotword.txt file in the model directory and add custom hotwords in the format of one Chinese word per line, such as adding industry-specific terms, specific personal names, and other hotword content.
        - Then, add relevant parameters in the code. The example is as follows:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline";
string modelFilePath = applicationBase + "./" + modelName + "/model.int8.onnx";
string modelebFilePath = applicationBase + "./" + modelName + "/model_eb.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string hotwordFilePath = applicationBase + "./" + modelName + "/hotword.txt";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath);
```
3. **Calling Process**
```csharp
List<float[]> samples = new List<float[]>();
// The code for converting the wav file into samples is omitted here. For details, refer to the example code in ManySpeech.AliParaformerAsr.Examples.
List<OfflineStream> streams = new List<OfflineStream>();
foreach (var sample in samples)
{
    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```
4. **Example of Output Results**
```
欢迎大家来体验达摩院推出的语音识别模型

非常的方便但是现在不同啊英国脱欧欧盟内部完善的产业链的红利人

he must be home now for the light is on他一定在家因为灯亮着就是有一种推理或者解释的那种感觉

elapsed_milliseconds: 1502.8828125
total_duration: 40525.6875
rtf: 0.037084696280599808
end!
```

### (B) Real-time (Streaming) Model Calling
1. **Adding Project References**
Add the following references in the code as well:
```csharp
using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
```
2. **Model Initialization and Configuration**
```csharp
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```
3. **Calling Process**
```csharp
List<float[]> samples = new List<float[]>();
// The code for converting the wav file into samples is omitted here. The following is the sample code for batch processing:
List<OnlineStream> streams = new List<OnlineStream>();
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
foreach (var sample in samples)
{
    OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
// Example of single processing. Only one stream needs to be constructed.
OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
// Refer to the example code in ManySpeech.AliParaformerAsr.Examples for details.
```
4. **Example of Output Results**
```

正是因为存在绝对正义所以我我接受现实式相对生但是不要因因现实的相对对正义们就就认为这个世界有有证因为如果当你认为这这个界界

elapsed_milliseconds: 1389.3125
total_duration: 13052
rtf: 0.10644441464909593
Hello, World!
```

## V. Related Projects
- **Voice Activity Detection**: To solve the problem of reasonable segmentation of long audio, you can add the ManySpeech.AliFsmnVad library and install it with the following command:
```bash
dotnet add package ManySpeech.AliFsmnVad
```
- **Text Punctuation Prediction**: To address the lack of punctuation in recognition results, you can add the ManySpeech.AliCTTransformerPunc library. The installation command is as follows:
```bash
dotnet add package ManySpeech.AliCTTransformerPunc
```
Specific calling examples can be referred to in the official documentation of the corresponding libraries or the `ManySpeech.AliParaformerAsr.Examples` project. This project is a console/desktop example project mainly used to demonstrate the basic functions of speech recognition, such as offline transcription and real-time recognition.

## VI. Other Notes
- **Test Cases**: Use `ManySpeech.AliParaformerAsr.Examples` as the test case.
- **Test CPU**: The test CPU used is Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (2.59 GHz).
- **Supported Platforms**:
    - **Windows**: Windows 7 SP1 and higher versions.
    - **macOS**: macOS 10.13 (High Sierra) and higher versions, and also supports iOS, etc.
    - **Linux**: Applicable to Linux distributions, but specific dependencies need to be met (see the list of Linux distributions supported by.NET 6 for details).
    - **Android**: Supports Android 5.0 (API 21) and higher versions.

## VII. Model Download (Supported ONNX Models)
The following is the information related to the ONNX models supported by ManySpeech.AliParaformerAsr, including model names, types, supported languages, punctuation status, timestamp status, and download addresses, which facilitates you to choose the appropriate model for download and use according to specific requirements:

| Model Name | Type | Supported Languages | Punctuation | Timestamp | Download Address |
| ---- | ---- | ---- | ---- | ---- | ---- |
| paraformer-large-zh-en-onnx-offline | Non-streaming | Chinese, English | No | No | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"), [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-timestamp-onnx-offline | Non-streaming | Chinese, English | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline "modelscope") |
| paraformer-large-en-onnx-offline | Non-streaming | English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-onnx-online | Streaming | Chinese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online "modelscope") |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 | Streaming | Chinese, Cantonese, English | No | No | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
| paraformer-seaco-large-zh-timestamp-onnx-offline | Non-streaming | Chinese, Hotwords | No | Yes | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
| SenseVoiceSmall | Non-streaming | Chinese, Cantonese, English, Japanese, Korean | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |
| sensevoice-small-wenetspeech-yue-int8-onnx | Non-streaming | Cantonese, Chinese, English, Japanese, Korean | Yes | No | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx "modelscope") |

## VIII. Model Introduction

### (A) Model Usage
Paraformer is an efficient non-autoregressive end-to-end speech recognition framework proposed by the speech team of DAMO Academy. The Paraformer Chinese general-purpose speech recognition model in this project is trained with tens of thousands of hours of labeled audio in the industrial field, which endows the model with good general recognition performance. It can be widely applied in scenarios such as speech input methods, speech navigation, and intelligent meeting minutes, and has a relatively high recognition accuracy.

### (B) Model Structure
The Paraformer model structure mainly consists of five parts: Encoder, Predictor, Sampler, Decoder, and Loss function. You can view its structural diagram [here](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true). The specific functions of each part are as follows:
- **Encoder**: It can adopt different network structures, such as self-attention, conformer, SAN-M, etc., and is mainly responsible for extracting acoustic features from audio.
- **Predictor**: It is a two-layer FFN (Feed Forward Neural Network). Its function is to predict the number of target words and extract the acoustic vectors corresponding to the target words, providing key data for subsequent recognition processing.
- **Sampler**: It is a module without learnable parameters. It can generate semantic feature vectors based on the input acoustic vectors and target vectors, enriching the semantic information for recognition.
- **Decoder**: Its structure is similar to that of the autoregressive model, but it is a bidirectional modeling (while the autoregressive model is unidirectional modeling). Through the bidirectional structure, it can better model the context and improve the accuracy of speech recognition.
- **Loss function**: Besides including the Cross Entropy (CE) and Minimum Word Error Rate (MWER) as discriminative optimization objectives, it also covers the Predictor optimization objective Mean Absolute Error (MAE). These optimization objectives ensure the accuracy of the model.

### (C) Main Highlights
- **Predictor Module**: Based on the Continuous integrate-and-fire (CIF) predictor, it extracts the acoustic feature vectors corresponding to the target words. In this way, it can predict the number of target words in the speech more accurately and improve the accuracy of speech recognition.
- **Sampler**: Through the sampling operation, it transforms the acoustic feature vectors and target word vectors into semantic feature vectors. Then, in cooperation with the bidirectional Decoder, it can significantly enhance the model's ability to understand and model the context, making the recognition results more in line with semantic logic.
- **MWER Training Criterion Based on Negative Sample Sampling**: This training criterion helps the model optimize parameters better during the training process, reduces recognition errors, and improves the overall recognition performance.

### (D) More Detailed Information
- **Model Links**:
    - [paraformer-large-offline (Non-streaming)](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline (Non-streaming)")
    - [paraformer-large-online (Streaming)](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online (Streaming)")
    - [SenseVoiceSmall (Non-streaming)](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall (Non-streaming)")
- **Paper**: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")
- **Paper Interpretation**: [Paraformer: High Recognition Rate and High Computational Efficiency Single-round Non-autoregressive End-to-End Speech Recognition Model](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: High Recognition Rate and High Computational Efficiency Single-round Non-autoregressive End-to-End Speech Recognition Model")

**Reference**
[1] https://github.com/alibaba-damo-academy/FunASR