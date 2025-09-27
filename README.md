（[简体中文](README.zh_CN.md) |  English）

 <div style="text-align:center;"><img width="128" height="128" align="center" alt="dotnet_bot_128x128" src="https://github.com/user-attachments/assets/d8119208-17a5-4bff-b222-003093ad7d18" /><div>

# ManySpeech Voice Processing Suite: A Cross-Platform C# Voice Solution 🎤

## Project Overview 📋

ManySpeech ( https://github.com/manyeyes/ManySpeech ) is a voice processing suite developed by the manyeyes community, which is based on C#. With excellent open-source models as its core, it relies on Microsoft.ML.OnnxRuntime to implement ONNX model decoding. It aims to solve three key issues:
- 🚩 Compatibility issues in cross-platform deployment
- 🚩 Model adaptation challenges in different scenarios (real-time/offline, multi-language)
- 🚩 The integration threshold of complex toolchains

As a solution that balances "usability, functionality, and deployment flexibility", ManySpeech can effectively improve development efficiency and provide strong support for voice processing needs in the .NET ecosystem.

## Core Features 🌟

### 1. Meeting the Actual Needs of C# Developers 👨💻

Starting from the actual scenarios of C# development, ManySpeech highly conforms to the .NET ecosystem in terms of model coverage, platform compatibility, and development process, and can be used as an important reference for tool selection.

### 2. Multi-Scenario Model Coverage 🧩

ManySpeech realizes the coordination of multiple tasks such as **"voice recognition, endpoint detection, punctuation restoration, audio separation and enhancement"**. There is no need to integrate multiple sets of toolchains—different business needs can be met simply by combining components, thus solving the problem of "tool fragmentation caused by variable scenarios".

### 3. Full Platform Compatibility and Reducing the Cost of Multi-End Deployment 🌐

Although C# already has a mature cross-platform capability relying on the .NET ecosystem, voice processing still faces many challenges due to its deep dependence on underlying resources (such as ONNX runtime, audio interfaces). ManySpeech effectively alleviates these cross-platform pain points:
- **Wide Framework Support** 🛠️: Compatible with .NET 4.6.1+, .NET 6.0+, .NET Core 3.1, .NET Standard 2.0+, covering mainstream frameworks from traditional desktop development to cross-platform development.
- **Comprehensive System Adaptation** 💻📱: Runs on Windows 7+, macOS 10.13+, all Linux distributions (in line with the .NET 6 support list), Android 5.0+, and iOS. Stable deployment is available for desktop software, server batch processing, and mobile apps.
- **Lightweight Optimization** ⚡: Supports AOT compilation—after compilation, the executable file volume is reduced by 30%+ and startup speed is increased by about 20%, suitable for embedded devices (e.g., IoT voice control terminals) and lightweight deployment scenarios.

### 4. Adaptation to Multiple Tasks, Unified Component Specifications, and Flexible Collaboration 🤝

#### (1) Voice Recognition Task 🎙️→📝
**Core Function**: Transcribing voice into text.
- Streaming models: Low latency, suitable for real-time interaction (online customer service, voice input methods) ⚡
- Non-streaming models: Suitable for offline transcription (local recordings, video subtitles) 📹
- Multi-language support: Covers Chinese, English, Cantonese, Japanese, Korean (e.g., SenseVoice model) 🌍
- Precision features: Word-level (Chinese)/word-level (English) timestamps (millisecond precision) + custom hot words (e.g., paraformer-seaco-large-zh-timestamp-onnx-offline), adaptable to vertical fields like "financial terms" and "medical terms" 🔍
- Smart punctuation: SenseVoiceSmall & whisper-large models support multi-language recognition with built-in punctuation prediction, no manual punctuation needed ✏️

**Related Components**:
- ManySpeech.AliParaformerAsr: Supports ONNX models (Paraformer, SenseVoice)
- ManySpeech.FireRedAsr: Supports ONNX models (FireRedASR-AED-L)
- ManySpeech.K2TransducerAsr: Supports ONNX models (Zipformer in the new Kaldi)
- ManySpeech.MoonshineAsr: Supports ONNX models (Tiny, Base in Moonshine)
- ManySpeech.WenetAsr: Supports ONNX models in WeNet
- ManySpeech.WhisperAsr: Supports ONNX models of the whisper series (voice language recognition)

#### (2) Voice Endpoint Detection Task 🎯
**Core Function**: Accurately detecting the start and end time points of valid voice in long voice segments.
- Reduces recognition errors from invalid voice by extracting valid audio segments for the recognition engine 📊
- Applicable scenarios: Offline transcription of long voice (conference recording processing) + real-time voice interaction (intelligent customer service response) 🗣️

**Related Components**:
- ManySpeech.AliFsmnVad: Supports Fsmn-Vad model (focuses on valid voice time point detection)
- ManySpeech.SileroVad: Supports Silero-VAD model (core function: valid voice time point detection in long segments)

#### (3) Punctuation Restoration Task ✏️
**Core Function**: Automatically predicting and adding punctuation to text (especially output from voice recognition models).
- Optimizes text structure via post-processing to improve readability and coherence 📄

**Related Components**:
- ManySpeech.AliCTTransformerPunc: Supports CT-Transformer model (punctuation prediction for voice recognition text output)

#### (4) Sound Source Separation & Voice Enhancement Task 🎧
**Core Function**: Focuses on audio separation, noise reduction, and enhancement.
- Separates target sounds from mixed audio (e.g., human voice from background sound) 🚫🔊
- Improves target voice clarity for high-quality input in subsequent tasks (voice recognition, audio processing) 📈
- Applicable scenarios: Conference recording noise reduction, voice extraction in noisy environments, multi-speaker audio separation 🎤👥

**Related Components**:
- ManySpeech.AudioSep: Supports ONNX models (clearervoice, gtcrn, spleeter, uvr) (focuses on audio separation, noise reduction, enhancement)

## Quick Start 🚀

Check our **"sample programs"** to learn how to use ManySpeech for application development.

## Supporters 🙏

Thank you to all our supporters! Your support drives the continuous improvement of ManySpeech.

## ManySpeech - Developing AI Voice Applications with C# 🤖

ManySpeech will continue to integrate cutting-edge AI models, providing developers with a low-threshold enterprise-level voice processing integration path. Developers can choose the most suitable scheme based on project needs (e.g., real-time response, dialect support) and model characteristics to efficiently promote function implementation.
