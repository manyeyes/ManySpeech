// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace ManySpeech.AudioSep.Utils
{
    // 源生成器的上下文配置
    [JsonSourceGenerationOptions(WriteIndented = true)] // 配置序列化选项
    [JsonSerializable(typeof(Model.ConfEntity))] // 指定需要序列化的类型
    public partial class AudioSepJsonContext : JsonSerializerContext
    {
        // 生成器会自动填充实现
    }
    /// <summary>
    /// YamlHelper
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    internal class PreloadHelper
    {
        public static T? ReadYaml<T>(string yamlFilePath)
        {
            T? info = default(T);
            IDeserializer yamlDeserializer = new StaticDeserializerBuilder(new YamlStaticContext()).WithNamingConvention(UnderscoredNamingConvention.Instance).Build();
            if (!string.IsNullOrEmpty(yamlFilePath) && yamlFilePath.IndexOf("/") < 0 && yamlFilePath.IndexOf("\\") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(yamlFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{yamlFilePath}' not found.");
                using (var yamlReader = new StreamReader(stream))
                {
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
            else if (File.Exists(yamlFilePath))
            {
                using (var yamlReader = File.OpenText(yamlFilePath))
                {
                    info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
            return info;
        }
        /// <summary>
        /// ReadJson for ConfEntity (To compile for AOT)
        /// </summary>
        /// <param name="jsonFilePath"></param>
        /// <returns></returns>
        /// <exception cref="FileNotFoundException"></exception>
        public static Model.ConfEntity? ReadJson(string jsonFilePath)
        {
            Model.ConfEntity? info = new Model.ConfEntity();
            if (!string.IsNullOrEmpty(jsonFilePath) && jsonFilePath.IndexOf("/") < 0 && jsonFilePath.IndexOf("\\") < 0)
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(jsonFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{jsonFilePath}' not found.");
                using (var jsonReader = new StreamReader(stream))
                {
                    info = JsonSerializer.Deserialize(jsonReader.ReadToEnd(), AudioSepJsonContext.Default.ConfEntity);
                    jsonReader.Close();
                }
            }
            else if (File.Exists(jsonFilePath))
            {
                using (var jsonReader = File.OpenText(jsonFilePath))
                {
                    info = JsonSerializer.Deserialize(jsonReader.ReadToEnd(), AudioSepJsonContext.Default.ConfEntity);
                    jsonReader.Close();
                }
            }
            return info;
        }
    }
}
