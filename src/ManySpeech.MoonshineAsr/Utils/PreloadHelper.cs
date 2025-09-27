// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
//using System.Reflection;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
//using YamlDotNet.Serialization;
//using YamlDotNet.Serialization.NamingConventions;

namespace ManySpeech.MoonshineAsr.Utils
{
    // 源生成器的上下文配置
    [JsonSourceGenerationOptions(WriteIndented = true)] // 配置序列化选项
    [JsonSerializable(typeof(Model.CustomMetadata))] // 指定需要序列化的类型
    public partial class AppJsonContext : JsonSerializerContext
    {
        // 生成器会自动填充实现
    }
    // 源生成器的上下文配置
    //[YamlStaticContext]
    //[YamlSerializable(typeof(Model.CustomMetadata))] // 指定需要序列化的类型
    //public partial class YamlStaticContext : YamlDotNet.Serialization.StaticContext
    //{
    //    // 生成器会自动填充实现
    //}

    /// <summary>
    /// YamlHelper
    /// Copyright (c)  2024 by manyeyes
    /// </summary>
    internal class PreloadHelper
    {
        public static T? ReadYaml<T>(string yamlFilePath)
        {
            T? info = default(T);
            ////Deserializer yamlDeserializer = new Deserializer();
            //IDeserializer yamlDeserializer = new StaticDeserializerBuilder(new YamlStaticContext()).WithNamingConvention(UnderscoredNamingConvention.Instance).Build();
            if (!string.IsNullOrEmpty(yamlFilePath) && yamlFilePath.IndexOf("/") < 0 && yamlFilePath.IndexOf("\\") < 0)
            {
                //var assembly = Assembly.GetExecutingAssembly();
                var assembly = typeof(PreloadHelper).Assembly;
                var stream = assembly.GetManifestResourceStream(yamlFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{yamlFilePath}' not found.");
                using (var yamlReader = new StreamReader(stream))
                {
                    //info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
            else if (File.Exists(yamlFilePath))
            {
                using (var yamlReader = File.OpenText(yamlFilePath))
                {
                    //info = yamlDeserializer.Deserialize<T>(yamlReader);
                    yamlReader.Close();
                }
            }
#pragma warning disable CS8603 // 可能返回 null 引用。
            return info;
#pragma warning restore CS8603 // 可能返回 null 引用。
        }

        public static T? ReadJson<T>(string jsonFilePath)
        {
            T? info = default(T);
            if (!string.IsNullOrEmpty(jsonFilePath) && jsonFilePath.IndexOf("/") < 0 && jsonFilePath.IndexOf("\\") < 0)
            {
                //var assembly = Assembly.GetExecutingAssembly();
                var assembly = typeof(PreloadHelper).Assembly;
                var stream = assembly.GetManifestResourceStream(jsonFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{jsonFilePath}' not found.");
                using (var jsonReader = new StreamReader(stream))
                {
                    info = JsonSerializer.Deserialize<T>(jsonReader.ReadToEnd());
                    jsonReader.Close();
                }
            }
            else if (File.Exists(jsonFilePath))
            {
                using (var jsonReader = File.OpenText(jsonFilePath))
                {
                    info = JsonSerializer.Deserialize<T>(jsonReader.ReadToEnd());
                    jsonReader.Close();
                }
            }
            return info;
        }
        /// <summary>
        /// ReadJson for CustomMetadata (To compile for AOT)
        /// </summary>
        /// <param name="jsonFilePath"></param>
        /// <returns></returns>
        /// <exception cref="FileNotFoundException"></exception>
        public static Model.CustomMetadata? ReadJson(string jsonFilePath)
        {
            Model.CustomMetadata? info = new Model.CustomMetadata();
            if (!string.IsNullOrEmpty(jsonFilePath) && jsonFilePath.IndexOf("/") < 0 && jsonFilePath.IndexOf("\\") < 0)
            {
                //var assembly = Assembly.GetExecutingAssembly();
                var assembly = typeof(PreloadHelper).Assembly;
                var stream = assembly.GetManifestResourceStream(jsonFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{jsonFilePath}' not found.");
                using (var jsonReader = new StreamReader(stream))
                {
                    info = JsonSerializer.Deserialize(jsonReader.ReadToEnd(), AppJsonContext.Default.CustomMetadata);
                    jsonReader.Close();
                }
            }
            else if (File.Exists(jsonFilePath))
            {
                using (var jsonReader = File.OpenText(jsonFilePath))
                {
                    info = JsonSerializer.Deserialize(jsonReader.ReadToEnd(), AppJsonContext.Default.CustomMetadata);
                    jsonReader.Close();
                }
            }
            return info;
        }

        public static string[] ReadTokens(string tokensFilePath)
        {
            string[] tokens = null;
            if (!string.IsNullOrEmpty(tokensFilePath) && tokensFilePath.IndexOf("/") < 0 && tokensFilePath.IndexOf("\\") < 0)
            {
                //var assembly = Assembly.GetExecutingAssembly();
                var assembly = typeof(PreloadHelper).Assembly;
                var stream = assembly.GetManifestResourceStream(tokensFilePath) ??
                             throw new FileNotFoundException($"Embedded resource '{tokensFilePath}' not found.");
                using (var reader = new StreamReader(stream))
                {
                    tokens = reader.ReadToEnd().Split('\n');
                }
            }
            else
            {
                tokens = File.ReadAllLines(tokensFilePath);
            }
            return tokens;
        }
    }
}
