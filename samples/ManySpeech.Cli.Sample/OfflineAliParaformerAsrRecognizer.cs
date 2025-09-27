using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Cli.Sample
{
    internal partial class AliParaformerAsrRecognizer : BaseAsr
    {
        private static OfflineRecognizer? _offlineRecognizer;
        public static OfflineRecognizer InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_offlineRecognizer == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string modelFilePath = modelBasePath + "./" + modelName + "/model.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/asr.yaml";
                string mvnFilePath = modelBasePath + "./" + modelName + "/am.mvn";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
                string modelebFilePath = modelBasePath + "./" + modelName + "/model_eb.int8.onnx";
                string hotwordFilePath = modelBasePath + "./" + modelName + "/hotword.txt";
                try
                {
                    string folderPath = Path.Combine(modelBasePath, modelName);
                    // 1. Check if the folder exists
                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine($"Error: folder does not exist - {folderPath}");
                        return null;
                    }
                    // 2. Obtain the file names and destination paths of all files
                    // (calculate the paths in advance to avoid duplicate concatenation)
                    var fileInfos = Directory.GetFiles(folderPath)
                        .Select(filePath => new
                        {
                            FileName = Path.GetFileName(filePath),
                            // Recommend using Path. Combine to handle paths (automatically adapt system separators)
                            TargetPath = Path.Combine(modelBasePath, modelName, Path.GetFileName(filePath))
                            // If it is necessary to strictly maintain the original splicing method, it can be replaced with:
                            // TargetPath = $"{modelBasePath}/./{modelName}/{Path.GetFileName(filePath)}"
                        })
                        .ToList();

                    // Process model path (priority: containing modelAccuracy>last one that matches prefix)
                    var modelCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model") && !f.FileName.Contains("_eb"))
                        .ToList();
                    if (modelCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredModel = modelCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        modelFilePath = preferredModel?.TargetPath ?? modelCandidates.Last().TargetPath;
                    }

                    // Process modeleb path
                    var modelebCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model_eb"))
                        .ToList();
                    if (modelebCandidates.Any())
                    {
                        var preferredModeleb = modelebCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        modelebFilePath = preferredModeleb?.TargetPath ?? modelebCandidates.Last().TargetPath;
                    }

                    // Process config paths (take the last one that matches the prefix)
                    configFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("asr") && (f.FileName.EndsWith(".yaml") || f.FileName.EndsWith(".json")))
                        ?.TargetPath ?? "";

                    // Process mvn paths (take the last one that matches the prefix)
                    mvnFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("am") && f.FileName.EndsWith(".mvn"))
                        ?.TargetPath ?? "";

                    // Process token paths (take the last one that matches the prefix)
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("tokens") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    // Process hotword paths (take the last one that matches the prefix)
                    hotwordFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("hotword") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    if (string.IsNullOrEmpty(modelFilePath) || string.IsNullOrEmpty(tokensFilePath))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath, threadsNum: threadsNum);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds_init = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds_init.ToString());
                }
                catch (UnauthorizedAccessException)
                {
                    Console.WriteLine($"Error: No permission to access this folder");
                }
                catch (PathTooLongException)
                {
                    Console.WriteLine($"Error: File path too long");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error occurred: {ex}");
                }
            }
            return _offlineRecognizer;
        }
        public static void OfflineRecognizer(string streamDecodeMethod = "one", string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null, string? modelBasePath = null)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            OfflineRecognizer offlineRecognizer = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (offlineRecognizer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            TimeSpan total_duration = new TimeSpan(0L);
            List<float[]>? samples = new List<float[]>();
            List<string> paths = new List<string>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                //mediaFilePaths = Directory.GetFiles(Path.Combine(modelBasePath, modelName, "test_wavs"));
                string fullPath = Path.Combine(modelBasePath, modelName);
                if (!Directory.Exists(fullPath))
                {
                    mediaFilePaths = Array.Empty<string>(); // 路径不正确时返回空数组
                }
                else
                {
                    mediaFilePaths = Directory.GetFiles(
                        path: fullPath,
                        searchPattern: "*.wav",
                        searchOption: SearchOption.AllDirectories
                    );
                }
            }
            foreach (string mediaFilePath in mediaFilePaths)
            {
                if (!File.Exists(mediaFilePath))
                {
                    continue;
                }
                if (AudioHelper.IsAudioByHeader(mediaFilePath))
                {
                    TimeSpan duration = TimeSpan.Zero;
                    float[]? sample = AudioHelper.GetFileSample(wavFilePath: mediaFilePath, duration: ref duration);
                    if (sample != null)
                    {
                        paths.Add(mediaFilePath);
                        samples.Add(sample);
                        total_duration += duration;
                    }
                }
            }
            if (samples.Count == 0)
            {
                Console.WriteLine("No media file is read!");
                return;
            }
            Console.WriteLine("Automatic speech recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "batch" : streamDecodeMethod;//one ,batch
            if (streamDecodeMethod == "one")
            {
                // Non batch method
                Console.WriteLine("Recognition results:\r\n");
                try
                {
                    int n = 0;
                    foreach (var sample in samples)
                    {
                        OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                        // Modify the logic here to dynamically modify hot words
                        //stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(modelBasePath, modelName, "tokens.txt"), new string[] {"魔搭" });
                        stream.AddSamples(sample);
                        OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                        Console.WriteLine($"{paths[n]}");
                        StringBuilder r = new StringBuilder();
                        r.Append("{");
                        r.Append($"\"text\": \"{result.Text}\",");
                        r.Append($"\"tokens\":[{string.Join(",", result.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                        r.Append($"\"timestamps\":[{string.Join(",", result.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                        r.Append("}");
                        Console.WriteLine($"{r.ToString()}");
                        Console.WriteLine("");
                        n++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    Console.WriteLine(ex.InnerException?.InnerException);
                }
                // Non batch method
            }
            if (streamDecodeMethod == "batch")
            {
                //2. batch method
                Console.WriteLine("Recognition results:\r\n");
                try
                {
                    int n = 0;
                    List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
                    foreach (var sample in samples)
                    {
                        AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                        stream.AddSamples(sample);
                        streams.Add(stream);
                    }
                    List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                    foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
                    {
                        Console.WriteLine($"{paths[n]}");
                        StringBuilder r = new StringBuilder();
                        r.Append("{");
                        r.Append($"\"text\": \"{result.Text}\",");
                        r.Append($"\"tokens\":[{string.Join(",", result.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                        r.Append($"\"timestamps\":[{string.Join(",", result.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                        r.Append("}");
                        Console.WriteLine($"{r.ToString()}");
                        Console.WriteLine("");
                        n++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    Console.WriteLine(ex.InnerException?.InnerException.Message);
                }
            }
            if (_offlineRecognizer != null)
            {
                _offlineRecognizer.Dispose();
                _offlineRecognizer = null;
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("recognition_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration_milliseconds:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }
    }
}
