# FireRedAsr
FireRedAsr is a C# library for decoding the FireRedASR AED-L model, used in speech recognition (ASR).

##### Introduction:
FireRedAsr is a speech recognition (ASR) library developed based on C#. Boasting excellent compatibility, this library supports framework versions such as .NET 4.6.1+ and .NET 6.0+. It not only focuses on efficiently performing speech recognition tasks but also features outstanding cross-platform capabilities, supporting compilation and invocation on multiple platforms including Windows, Linux, Android, macOS, and iOS, making it widely adaptable to various scenarios.

The core-dependent FireRedASR-AED model is designed to balance high performance and computational efficiency. It adopts an attention-based encoder-decoder (AED) architecture and can serve as an efficient speech representation module in large language model (LLM)-based speech models, providing stable and high-quality technical support for speech recognition tasks.

##### Supported Models (ONNX)
| Model Name  | Type | Supported Languages  | Download Link  |
| ------------ | ------------ | ------------ | ------------ |
| fireredasr-aed-large-zh-en-onnx-offline-20250124 | Non-streaming  | Chinese, English  |[modelscope](https://www.modelscope.cn/models/manyeyes/fireredasr-aed-large-zh-en-onnx-offline-20250124 "modelscope") |

##### How to Use
###### 1. Clone the project source code
```bash
cd /path/to
git clone https://github.com/manyeyes/FireRedASR.git
```
###### 2. Download the model from the above list to the directory: /path/to/FireRedASR/FireRedASR.Examples
```bash
cd /path/to/FireRedASR/FireRedASR.Examples
git clone https://www.modelscope.cn/manyeyes/[model name].git
```
###### 3. Load the project using VS2022 (or other IDEs)
###### 4. Set the files in the model directory to: Copy to Output Directory -> Copy if newer
###### 5. Modify the code in the example: string modelName = [model directory name]
###### 6. Run the project
###### 7. How to Invoke
Refer to the sample code in FireRedAsrExamples.cs
###### 8. Running Results
```
朱立南在上市见面会上表示

这是第一种第二种叫呃与always always什么意思啊

好首先说一下刚才这个经理说完了这个销售问题咱再说一下咱们的商场问题首先咱们商场上半年业这个先各部门儿汇报一下就是业绩

elapsed_milliseconds:4391.234375
total_duration:21015.0625
rtf:0.2089565222563578
Hello, World!
```
###### Related Projects:
* Speech endpoint detection, solving the problem of reasonable segmentation of long audio. Project address: [AliFsmnVad](https://github.com/manyeyes/AliFsmnVad "AliFsmnVad") 
* Text punctuation prediction, solving the problem of missing punctuation in recognition results. Project address: [AliCTTransformerPunc](https://github.com/manyeyes/AliCTTransformerPunc "AliCTTransformerPunc")

###### Other Instructions:

Test case: FireRedASR.Examples.
Test CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
Supported platforms:
Windows 7 SP1 or later,
macOS 10.13 (High Sierra) or later, iOS, etc.,
Linux distributions (specific dependencies are required, see the list of Linux distributions supported by .NET 6),
Android (Android 5.0 (API 21) or later).

Reference
----------
[1] https://github.com/FireRedTeam/FireRedASR