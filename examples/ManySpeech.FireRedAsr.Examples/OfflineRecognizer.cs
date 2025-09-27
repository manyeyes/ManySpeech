using ManySpeech.FireRedAsr.Model;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.FireRedAsr.Examples
{
    internal static partial class Program
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
                string encoderFilePath = modelBasePath + "./" + modelName + "/encoder.int8.onnx";
                string decoderFilePath = modelBasePath + "./" + modelName + "/decoder.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/config.json";
                string mvnFilePath = modelBasePath + "./" + modelName + "/am.mvn";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
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

                    // Process encoder path (priority: containing modelAccuracy>last one that matches prefix)
                    var encoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model") && !f.FileName.Contains("_eb"))
                        .ToList();
                    if (encoderCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredEncoder = encoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        encoderFilePath = preferredEncoder?.TargetPath ?? encoderCandidates.Last().TargetPath;
                    }

                    // Process decoder path
                    var decoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model_eb"))
                        .ToList();
                    if (decoderCandidates.Any())
                    {
                        var preferredDecoder = decoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        decoderFilePath = preferredDecoder?.TargetPath ?? decoderCandidates.Last().TargetPath;
                    }

                    // Process config paths (take the last one that matches the prefix)
                    //configFilePath = fileInfos
                    //    .LastOrDefault(f => f.FileName.StartsWith("asr") && (f.FileName.EndsWith(".json")))
                    //    ?.TargetPath ?? "";

                    // Process mvn paths (take the last one that matches the prefix)
                    mvnFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("am") && f.FileName.EndsWith(".mvn"))
                        ?.TargetPath ?? "";

                    // Process token paths (take the last one that matches the prefix)
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("tokens") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    if (string.IsNullOrEmpty(encoderFilePath) || string.IsNullOrEmpty(decoderFilePath) || string.IsNullOrEmpty(tokensFilePath))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _offlineRecognizer = new OfflineRecognizer(encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, threadsNum: threadsNum);
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
            List<string> paths= new List<string>();
            if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
            {
                mediaFilePaths = Directory.GetFiles(Path.Combine(modelBasePath, modelName, "test_wavs"));
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
            streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "multi" : streamDecodeMethod;//one ,multi
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
                        stream.AddSamples(sample);
                        OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                        Console.WriteLine($"{paths[n]}");
                        StringBuilder r = new StringBuilder();
                        r.Append("{");
                        r.Append($"\"text\": \"{result.Text}\",");
                        r.Append($"\"tokens\":[{string.Join(",",result.Tokens.Select(x=>$"\"{x}\"").ToArray())}],");
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
            if (streamDecodeMethod == "multi")
            {
                //2. batch method
                Console.WriteLine("Recognition results:\r\n");
                try
                {
                    int n = 0;
                    List<FireRedAsr.OfflineStream> streams = new List<FireRedAsr.OfflineStream>();
                    foreach (var sample in samples)
                    {
                        FireRedAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                        stream.AddSamples(sample);
                        streams.Add(stream);
                    }
                    List<OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                    foreach (OfflineRecognizerResultEntity result in results)
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
