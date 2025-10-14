# ManySpeech.AliFsmnVad User Guide

ManySpeech.AliFsmnVad is a C# library for **Fsmn-Vad model decoding**, primarily designed for **Voice Activity Detection (VAD)** to accurately identify valid speech segments in audio.


## 1. Introduction
Developed in C#, ManySpeech.AliFsmnVad enables efficient decoding of ONNX-format Fsmn-Vad models by leveraging the `Microsoft.ML.OnnxRuntime` component. It offers the following core features:
- **Excellent Compatibility**: Supports frameworks including .NET Framework 4.6.1+ and .NET 6.0+. Compatible with cross-platform compilation (Windows, macOS, Linux, Android, iOS) and AOT compilation for flexible deployment.
- **High Performance**: The Real-Time Factor (RTF) for the full speech endpoint detection process is approximately 0.008, with processing speed far exceeding the requirements of real-time audio streams.
- **Focused Functionality**: As a 16kHz general-purpose VAD tool, it is based on the "FSMN-Monophone VAD" efficient model proposed by the Speech Team of DAMO Academy. It accurately detects the start and end timestamps of valid speech in long audio segments. By extracting valid audio segments for input to speech recognition engines, it significantly reduces recognition errors caused by invalid speech and improves the accuracy of speech recognition tasks.


## 2. Installation
Install via NuGet Package Manager (recommended):

### 2.1 Using Package Manager Console
Execute the following command in the "Package Manager Console" of Visual Studio:
```bash
Install-Package ManySpeech.AliFsmnVad
```

### 2.2 Using .NET CLI
Run the following command in the command line:
```bash
dotnet add package ManySpeech.AliFsmnVad
```

### 2.3 Manual Installation
Search for "ManySpeech.AliFsmnVad" in the NuGet Package Manager interface and click "Install".


## 3. VAD Common Parameter Adjustment Instructions
Parameter configurations refer to the `vad.yaml` file in the project. Core adjustable parameters are as follows (optimize based on actual scenarios):

### 3.1 max_end_silence_time
- **Function**: Time threshold to trigger end-point detection when continuous silence is detected at the end of audio.
- **Parameter Range**: 500ms ~ 6000ms
- **Default Value**: 800ms
- **Notes**: A too-low value may cause premature truncation of valid speech, while a too-high value retains excessive silence segments. Adjust based on audio scenarios (e.g., meeting recordings, single-person speech).

### 3.2 speech_noise_thres
- **Function**: Core threshold for distinguishing "speech/noise". If the value of "speech score - noise score" exceeds this threshold, the audio is判定 as speech.
- **Parameter Range**: (-1, 1)
- **Adjustment Logic**:
  - A value closer to -1 increases the probability of noise being mistakenly identified as speech (higher False Alarm (FA) rate);
  - A value closer to +1 increases the probability of valid speech being mistakenly identified as noise (higher Miss Probability (Pmiss) rate);
- **Recommendation**: Typically, select a balanced value based on the model's performance on long audio test sets in the target scenario.


## 4. Usage
The following is a complete usage example, including namespace import, initialization, core calls, and result retrieval.

### 4.1 Import Namespace
After installation, import the namespace at the top of the code file:
```csharp
using ManySpeech.AliFsmnVad;
using ManySpeech.AliFsmnVad.Model;
```

### 4.2 Model and Configuration Initialization
Prepare 3 core files in advance (model file, configuration file, mean normalization file), and specify the file paths and batch decoding parameters during initialization:
```csharp
// Get the application root directory (avoid hardcoding paths)
string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
string modelDirName = "speech_fsmn_vad_zh-cn-16k-common-onnx"; // Model folder name

// Concatenate full paths for model, configuration, and mean normalization files
string modelFilePath = Path.Combine(applicationBase, modelDirName, "model.onnx");
string configFilePath = Path.Combine(applicationBase, modelDirName, "vad.yaml");
string mvnFilePath = Path.Combine(applicationBase, modelDirName, "vad.mvn");

int batchSize = 2; // Batch decoding size (adjust based on hardware performance, recommended: 1~4)

// Initialize AliFsmnVad instance (load model and configure parameters)
AliFsmnVad aliFsmnVad = new AliFsmnVad(modelFilePath, configFilePath, mvnFilePath, batchSize);
```

### 4.3 Call Core Methods
Choose different calling methods based on the audio file size: one-time processing for small files, and step-by-step processing for large files to reduce memory usage.

#### Method 1: One-Time Processing for Small Files
Suitable for short-duration audio (e.g., < 5 minutes):
```csharp
// samples: Audio sample data (read in advance via libraries like NAudio, format: float[])
SegmentEntity[] segments = aliFsmnVad.GetSegments(samples);
```

#### Method 2: Step-by-Step Processing for Large Files
Suitable for long-duration audio (e.g., > 5 minutes). Read and process audio data in batches:
```csharp
// samples: Audio sample data (type: float[], can be read in batches)
SegmentEntity[] segments = aliFsmnVad.GetSegmentsByStep(samples);
```

### 4.4 Retrieve and Parse Results
The `SegmentEntity` array contains core information about each valid speech segment. Traverse the array to extract speech segments and timestamps:
```csharp
// Traverse all valid speech segments
foreach (SegmentEntity segment in segments)
{
    // segment.Waveform: Audio sample data of the VAD-split speech segment (float[])
    // segment.Segment: Timestamp corresponding to the speech segment (millisecond-level, format: [start time, end time])
    Console.WriteLine($"Valid Speech Segment: {segment.Segment[0]}ms ~ {segment.Segment[1]}ms");
}
```

#### Example Output
```text
load model and init config elapsed_milliseconds: 463.5390625
vad infer result:
[[70,2340], [2620,6200], [6480,23670], [23950,26250], [26780,28990], [29950,31430], [31750,37600], [38210,46900], [47310,49630], [49910,56460], [56740,59540], [59820,70450]]
elapsed_milliseconds: 662.796875
total_duration: 70470.625ms
rtf: 0.009405292985552491
```
- Timestamp Format: `[start time, end time]` (unit: milliseconds). For example, `[70,2340]` indicates a valid speech segment from 70ms to 2340ms, with silence/noise segments automatically filtered out.


## 5. Integration with Speech Recognition
Use `SegmentEntity.Waveform` as the input parameter to connect to mainstream speech recognition libraries for subsequent recognition tasks. Supported libraries include:
- AliParaformerAsr
- K2TransducerAsr
- SherpaOnnxSharp (call its `offlineRecognizer`-related methods)

For specific calling examples, refer to the official documentation of the corresponding library or the `ManySpeech.AliFsmnVad.Examples` test project.


## 6. Additional Notes
### 6.1 Test Cases
An independent test project `ManySpeech.AliFsmnVad.Examples` is provided, containing complete examples of audio reading, VAD detection, and result parsing for direct reference and debugging.

### 6.2 Supported Platforms
- Windows: Windows 7 SP1 or later
- macOS: macOS 10.13 (High Sierra) or later
- Linux: Linux distributions compatible with .NET 6.0+ (install dependencies in advance; see [.NET Official Documentation](https://learn.microsoft.com/en-us/dotnet/core/install/linux))
- Android: Android 5.0 (API 21) or later
- iOS: Develop with Xamarin or .NET MAUI; supports iOS 11.0 or later

## 7. Model Download
Download the official Fsmn-Vad model (16kHz general-purpose version) from the following platforms:
- Hugging Face: [manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx](https://huggingface.co/manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx)
- ModelScope: [manyeyes/alifsmnvad-onnx](https://www.modelscope.cn/models/manyeyes/alifsmnvad-onnx)

For details about the official model, refer to: [damo/speech_fsmn_vad_zh-cn-16k-common-onnx](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx)


## 8. References
- [FunASR Official Repository](https://github.com/modelscope/FunASR) (The Fsmn-Vad model is derived from this project)