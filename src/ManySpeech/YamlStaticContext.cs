using YamlDotNet.Serialization;

namespace ManySpeech
{
    [YamlStaticContext]
    [YamlSerializable(typeof(ASR.Model.ConfEntity))] // 指定需要序列化的类型
    [YamlSerializable(typeof(ASR.Model.FrontendConf))]
    [YamlSerializable(typeof(ASR.Model.ModelConf))]
    [YamlSerializable(typeof(ASR.Model.EncoderConf))]
    [YamlSerializable(typeof(ASR.Model.DecoderConf))]
    [YamlSerializable(typeof(ASR.Model.PredictorConf))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.ConfEntity))] // 指定需要序列化的类型
    [YamlSerializable(typeof(AliParaformerAsr.Model.FrontendConf))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.ModelConf))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.EncoderConf))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.DecoderConf))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.PredictorConf))]
    [YamlSerializable(typeof(AliFsmnVad.Model.ConfEntity))] 
    [YamlSerializable(typeof(AliFsmnVad.Model.FrontendConfEntity))]
    [YamlSerializable(typeof(AliFsmnVad.Model.EncoderConfEntity))]
    [YamlSerializable(typeof(AudioSep.Model.ConfEntity))]
    [YamlSerializable(typeof(AudioTagging.Model.ConfEntity))]
    [YamlSerializable(typeof(DolphinAsr.Model.ConfEntity))]
    [YamlSerializable(typeof(DolphinAsr.Model.EncoderConfig))]
    [YamlSerializable(typeof(DolphinAsr.Model.DecoderConfig))]
    [YamlSerializable(typeof(DolphinAsr.Model.PreprocessorConfig))]
    [YamlSerializable(typeof(DolphinAsr.Model.FrontendConfig))]
    [YamlSerializable(typeof(FireRedAsr.Model.ConfEntity))] 
    [YamlSerializable(typeof(FireRedAsr.Model.FrontendConf))]
    [YamlSerializable(typeof(FireRedAsr.Model.PreprocessorConf))]
    [YamlSerializable(typeof(MoonshineAsr.Model.CustomMetadata))] 
    [YamlSerializable(typeof(SileroVad.Model.ModelCustomMetadata))]
    [YamlSerializable(typeof(TextPunc.Model.ConfEntity))]
    [YamlSerializable(typeof(TextPunc.Model.ModelConfEntity))]
    [YamlSerializable(typeof(OmniAsr.Model.ConfEntity))]
    [YamlSerializable(typeof(TextPunc.Model.PunctuationConfEntity))]
    [YamlSerializable(typeof(WhisperAsr.Model.ConfEntity))] 
    [YamlSerializable(typeof(WhisperAsr.Model.ModelDimensions))]
    [YamlSerializable(typeof(WhisperAsr.Model.DecodingOptions))]
    public partial class YamlStaticContext : YamlDotNet.Serialization.StaticContext
    {
        // 生成器会自动填充实现
    }
}
