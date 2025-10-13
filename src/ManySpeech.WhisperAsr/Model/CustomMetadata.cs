// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace ManySpeech.WhisperAsr.Model
{
    public class CustomMetadata
    {
        //model metadata
        private string? _onnx_Infer;

        public string? onnx_Infer { get => _onnx_Infer; set => _onnx_Infer = value; }
        
    }
}
