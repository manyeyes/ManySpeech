using YamlDotNet.Serialization;

namespace ManySpeech
{
    [YamlStaticContext]
    [YamlSerializable(typeof(AliParaformerAsr.Model.ConfEntity))] // 指定需要序列化的类型
    [YamlSerializable(typeof(AliParaformerAsr.Model.FrontendConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.ModelConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.PreEncoderConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.EncoderConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.PostEncoderConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.DecoderConfEntity))]
    [YamlSerializable(typeof(AliParaformerAsr.Model.PredictorConfEntity))]
    [YamlSerializable(typeof(AliCTTransformerPunc.Model.ConfEntity))] 
    [YamlSerializable(typeof(AliCTTransformerPunc.Model.ModelConfEntity))]
    [YamlSerializable(typeof(AliCTTransformerPunc.Model.PunctuationConfEntity))]
    [YamlSerializable(typeof(AliFsmnVad.Model.ConfEntity))] 
    [YamlSerializable(typeof(AliFsmnVad.Model.FrontendConfEntity))]
    [YamlSerializable(typeof(AliFsmnVad.Model.EncoderConfEntity))]
    [YamlSerializable(typeof(MoonshineAsr.Model.CustomMetadata))] 
    [YamlSerializable(typeof(SileroVad.Model.ModelCustomMetadata))]  
    [YamlSerializable(typeof(AudioSep.Model.ConfEntity))] 
    [YamlSerializable(typeof(WhisperAsr.Model.ConfEntity))] 
    [YamlSerializable(typeof(WhisperAsr.Model.ModelDimensions))]
    [YamlSerializable(typeof(WhisperAsr.Model.DecodingOptions))]
    public partial class YamlStaticContext : YamlDotNet.Serialization.StaticContext
    {
        // 生成器会自动填充实现
    }
}
