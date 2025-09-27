# AliParaformerAsr

##### Introduction:
AliParaformerAsr is a **speech recognition** library written in C#. It uses Microsoft.ML.OnnxRuntime under the hood to decode ONNX models, supporting multiple environments including net461+, net60+, netcoreapp3.1, and netstandard2.0+. It enables cross-platform compilation and AOT compilation, with a simple and user-friendly usage.


##### Supported Models (ONNX)

| Model Name  | Type | Supported Languages  | Punctuation  | Timestamp  | Download Link  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| paraformer-large-zh-en-onnx-offline | Non-streaming  | Chinese, English  | No | No  | [huggingface](https://huggingface.co/manyeyes/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx "huggingface"),  [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-offline "modelscope") |
| paraformer-large-zh-en-timestamp-onnx-offline | Non-streaming  | Chinese, English  | No | Yes  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-timestamp-onnx-offline "modelscope") |
| paraformer-large-en-onnx-offline | Non-streaming | English | No  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-en-onnx-offline "modelscope")  |
| paraformer-large-zh-en-onnx-online | Streaming | Chinese, English | No  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-en-onnx-online "modelscope")  |
| paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No  | Yes  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805 "modelscope")  |
| paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 | Non-streaming | Chinese, Cantonese, English | No  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805 "modelscope") |
| paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 | Streaming | Chinese, Cantonese, English | No  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208 "modelscope") |
| paraformer-seaco-large-zh-timestamp-onnx-offline | Non-streaming | Chinese, Hotwords | No  | Yes  | [modelscope](https://www.modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline "modelscope") |
| SenseVoiceSmall | Non-streaming | Chinese, Cantonese, English, Japanese, Korean | Yes  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-onnx "modelscope"), [modelscope-split-embed](https://www.modelscope.cn/models/manyeyes/sensevoice-small-split-embed-onnx "modelscope-split-embed") |
| sensevoice-small-wenetspeech-yue-int8-onnx | Non-streaming | Cantonese, Chinese, English, Japanese, Korean | Yes  | No  | [modelscope](https://www.modelscope.cn/models/manyeyes/sensevoice-small-wenetspeech-yue-int8-onnx "modelscope") |


## How to Run the Sample Projects

###### 1. Clone the Project to Local
```bash
cd /path/to
git clone https://github.com/manyeyes/AliParaformerAsr.git
``` 

###### 2. Download Models from the Above List to a Local Directory (Optional)
```bash
cd /path/to/AliParaformerAsr/AliParaformerAsr.Examples
git clone https://www.modelscope.cn/manyeyes/[Model Name].git
```

###### 3. Load the Project with Visual Studio 2022 (or Other IDEs)
Open Visual Studio 2022, navigate to **File > Open > Project/Solution**, select the `AliParaformerAsr.sln` file in the cloned project root directory, and click **Open** to load the project.


###### 4. Run the AliParaformerAsr.Examples Project
`AliParaformerAsr.Examples` is a console/desktop sample project used to demonstrate basic speech recognition functions (e.g., offline transcription, real-time recognition).


###### 5. Run the MauiApp1 Project
`MauiApp1` is a cross-platform project developed with .NET MAUI, supporting speech recognition on devices such as Android, iOS, and Windows.


###### 6. Configuration Instructions (Reference: asr.yaml File)
Most of the asr.yaml configuration parameters for decoding do not need modification. The modifiable parameters are:
- `use_itn: true` (in the SenseVoiceSmall configuration, enabling this parameter enables inverse text normalization).


## How to Call in Code

### Offline (Non-streaming) Model Calling Method:

###### 1. Add Project Reference
```csharp
using AliParaformerAsr;
```

###### 2. Model Initialization and Configuration
Calling method for Paraformer models:
```csharp
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelName = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
string modelFilePath = applicationBase + "./" + modelName + "/model_quant.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
AliParaformerAsr.OfflineRecognizer offlineRecognizer = new OfflineRecognizer(modelFilePath, configFilePath, mvnFilePath, tokensFilePath);
```

Calling method for SeACo-Paraformer models:
1. Modify the `hotword.txt` file in the model directory and add custom hotwords (currently supports the format of "one Chinese word per line").
2. Add new parameters in the code: `modelebFilePath`, `hotwordFilePath`.
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

###### 3. Calling
```csharp
List<float[]> samples = new List<float[]>();
// The conversion of WAV files to samples is omitted here...
// For details, refer to the sample code in AliParaformerAsr.Examples
List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
foreach (var sample in samples)
{
    AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
```

###### 4. Output Result:
```
Welcome everyone to experience the speech recognition model launched by DAMO Academy

It is very convenient, but now things are different. After Brexit, the dividends from the well-developed industrial chain within the EU...

He must be home now for the light is on – it implies a kind of reasoning or explanatory context.

elapsed_milliseconds:1502.8828125
total_duration:40525.6875
rtf:0.037084696280599808
end!
```


## Real-time (Streaming) Model Calling Method:

###### 1. Add Project Reference
```csharp
using AliParaformerAsr;
```

###### 2. Model Initialization and Configuration
```csharp
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.int8.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.int8.onnx";
string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
string mvnFilePath = applicationBase + "./" + modelName + "/am.mvn";
string tokensFilePath = applicationBase + "./" + modelName + "/tokens.txt";
OnlineRecognizer onlineRecognizer = new OnlineRecognizer(encoderFilePath, decoderFilePath, configFilePath, mvnFilePath, tokensFilePath);
```

###### 3. Calling
```csharp
List<float[]> samples = new List<float[]>();
// The conversion of WAV files to samples is omitted here...
// Details are omitted here; the following is a sample code for batch processing:
List<AliParaformerAsr.OnlineStream> streams = new List<AliParaformerAsr.OnlineStream>();
AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
foreach (var sample in samples)
{
    AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
    stream.AddSamples(sample);
    streams.Add(stream);
}
List<AliParaformerAsr.OnlineRecognizerResultEntity> results = onlineRecognizer.GetResults(streams);
// For single processing, only one stream needs to be constructed:
AliParaformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
stream.AddSamples(sample);
AliParaformerAsr.OnlineRecognizerResultEntity result = onlineRecognizer.GetResult(stream);
// For details, refer to the sample code in AliParaformerAsr.Examples
```

###### 4. Output Result
```
It is precisely because absolute justice exists that I accept the relativity of reality. However, we should not believe that there is "justice" in this world just because of the relative justice in reality – because if you think that...

elapsed_milliseconds:1389.3125
total_duration:13052
rtf:0.10644441464909593
Hello, World!
```


###### Related Projects:
* Speech Endpoint Detection: Solves the problem of reasonable segmentation of long audio. Project address: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* Text Punctuation Prediction: Solves the problem of missing punctuation in recognition results. Project address: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")


###### Other Notes:

Test Case: AliParaformerAsr.Examples  
Test CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz  

Supported Platforms:
- Windows 7 SP1 or later
- macOS 10.13 (High Sierra) or later, iOS
- Linux distributions (specific dependencies required; see the list of Linux distributions supported by .NET 6)
- Android (Android 5.0 (API 21) or later)


## Model Introduction:

##### Model Purpose:
Paraformer is an efficient non-autoregressive end-to-end speech recognition framework proposed by the DAMO Academy Speech Team. This project is a general-purpose Chinese speech recognition model based on Paraformer, trained on industrial-grade annotated audio of tens of thousands of hours to ensure the model's general recognition performance. The model can be applied to scenarios such as speech input methods, speech navigation, and intelligent meeting minutes. Accuracy: High.

##### Model Structure:
![](https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master&FilePath=fig/struct.png&View=true)

As shown in the figure above, the Paraformer model structure consists of five parts: Encoder, Predictor, Sampler, Decoder, and Loss Function. The Encoder can adopt different network structures, such as self-attention, conformer, and SAN-M. The Predictor is a two-layer FFN that predicts the number of target words and extracts acoustic vectors corresponding to the target words. The Sampler is a module without trainable parameters that generates feature vectors containing semantic information based on the input acoustic vectors and target vectors. The Decoder structure is similar to that of autoregressive models but uses bidirectional modeling (autoregressive models use unidirectional modeling). For the Loss Function part, in addition to cross-entropy (CE) and MWER discriminative optimization objectives, it also includes the Predictor optimization objective MAE.

##### Key Core Points:
- Predictor Module: A Continuous Integrate-and-Fire (CIF)-based Predictor is used to extract acoustic feature vectors corresponding to target words, enabling more accurate prediction of the number of target words in speech.
- Sampler: Through sampling, acoustic feature vectors and target word vectors are transformed into feature vectors containing semantic information. Combined with the bidirectional Decoder, it enhances the model's ability to model context.
- MWER training criterion based on negative sample sampling.


##### More Detailed Resources:
* [paraformer-large-offline (Non-streaming)](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch "paraformer-large-offline (Non-streaming)")
* [paraformer-large-online (Streaming)](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online "paraformer-large-online (Streaming)")
* [SenseVoiceSmall (Non-streaming)](https://www.modelscope.cn/models/iic/SenseVoiceSmall "SenseVoiceSmall (Non-streaming)")
* Paper: [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317 "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition")
* Paper Interpretation: [Paraformer: A Single-turn Non-autoregressive End-to-End Speech Recognition Model with High Recognition Rate and Computational Efficiency](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw "Paraformer: A Single-turn Non-autoregressive End-to-End Speech Recognition Model with High Recognition Rate and Computational Efficiency")


Reference
----------
[1] https://github.com/alibaba-damo-academy/FunASR