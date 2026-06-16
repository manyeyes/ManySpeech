using ManySpeech.ASR.Examples.Base;
using ManySpeech.ASR.Examples.Entities;
using ManySpeech.ASR.Model;
using PreProcessUtils;

namespace ManySpeech.ASR.Examples
{
    internal partial class OfflineASR : BaseASR
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
                string encoderFilePath = modelBasePath + "./" + modelName + "/encoder.int8.onnx";
                string adaptorFilePath = modelBasePath + "./" + modelName + "/adaptor.int8.onnx";
                string decoderFilePath = modelBasePath + "./" + modelName + "/decoder.int8.onnx";
                string embedFilePath = modelBasePath + "./" + modelName + "/embed.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/asr.yaml";
                string mvnFilePath = modelBasePath + "./" + modelName + "/am.mvn";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
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
                    modelFilePath = "";
                    if (modelCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredModel = modelCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        modelFilePath = preferredModel?.TargetPath ?? modelCandidates.Last().TargetPath;
                    }

                    // Process encoder path (priority: containing encoderAccuracy>last one that matches prefix)
                    var encoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("encoder"))
                        .ToList();
                    encoderFilePath = "";
                    if (encoderCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredEncoder = encoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        encoderFilePath = preferredEncoder?.TargetPath ?? encoderCandidates.Last().TargetPath;
                    }

                    // Process adaptor path (priority: containing adaptorAccuracy>last one that matches prefix)
                    var adaptorCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("adaptor"))
                        .ToList();
                    adaptorFilePath = "";
                    if (adaptorCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredAdaptor = adaptorCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        adaptorFilePath = preferredAdaptor?.TargetPath ?? adaptorCandidates.Last().TargetPath;
                    }

                    // Process decoder path (priority: containing modelAccuracy>last one that matches prefix)
                    var decoderCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("decoder"))
                        .ToList();
                    decoderFilePath = "";
                    if (decoderCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredDecoder = decoderCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        decoderFilePath = preferredDecoder?.TargetPath ?? decoderCandidates.Last().TargetPath;
                    }

                    // Process embed path
                    var embedCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("model_eb") || f.FileName.StartsWith("embed"))
                        .ToList();
                    embedFilePath = "";
                    if (embedCandidates.Any())
                    {
                        var preferredModeleb = embedCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        embedFilePath = preferredModeleb?.TargetPath ?? embedCandidates.Last().TargetPath;
                    }

                    // Process config paths (take the last one that matches the prefix)
                    configFilePath = fileInfos
                        .LastOrDefault(f => (f.FileName.StartsWith("conf.") || f.FileName.StartsWith("asr.")) && (f.FileName.EndsWith(".yaml") || f.FileName.EndsWith(".json")))
                        ?.TargetPath ?? "";

                    // Process mvn paths (take the last one that matches the prefix)
                    mvnFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("am") && f.FileName.EndsWith(".mvn"))
                        ?.TargetPath ?? "";

                    // Process token paths (take the last one that matches the prefix)
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.EndsWith(".tiktoken"))
                        ?.TargetPath ?? "";
                    if (string.IsNullOrEmpty(tokensFilePath))
                    {
                        tokensFilePath = fileInfos
                            .LastOrDefault(f => f.FileName.StartsWith("tokens") && f.FileName.EndsWith(".txt"))
                            ?.TargetPath ?? "";
                    }

                    // Process hotword paths (take the last one that matches the prefix)
                    hotwordFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("hotword") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    if (string.IsNullOrEmpty(configFilePath) || string.IsNullOrEmpty(tokensFilePath))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _offlineRecognizer = new OfflineRecognizer(modelFilePath: modelFilePath, encoderFilePath: encoderFilePath, decoderFilePath: decoderFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, embedFilePath: embedFilePath, adaptorFilePath: adaptorFilePath, hotwordFilePath: hotwordFilePath, threadsNum: threadsNum);
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
        public static void Recognize(string streamDecodeMethod = "one", string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline", string modelAccuracy = "int8", int threadsNum = 2, string[]? mediaFilePaths = null, string[]? languages = null, string[]? hotwords = null, string? modelBasePath = null)
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
            if (languages == null)
                languages = new string[] { "中文" };
            DateTime processStartTime = DateTime.Now;
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
                        //This is a test,
                        //please set hot words and language according to actual business needs
                        if (hotwords == null)
                            stream.Hotwords = new List<string> { "魔搭", "开放时间" };
                        else
                            stream.Hotwords = hotwords.ToList();
                        if (languages.Length - 1 >= n)
                            stream.Language = languages[n];
                        else
                            stream.Language = languages.LastOrDefault();
                        stream.AddSamples(sample);
                        OfflineRecognizerResultEntity nativeResult = offlineRecognizer.GetResult(stream);
                        var processingTime = (DateTime.Now - processStartTime).TotalMilliseconds;
                        var resultEntity = ConvertToResultEntity(nativeResult, n, processingTime);
                        RaiseRecognitionResult(resultEntity);
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
                    int m = 0;
                    List<OfflineStream> streams = new List<OfflineStream>();
                    foreach (var sample in samples)
                    {
                        OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                        //This is a test,
                        //please set hot words and language according to actual business needs
                        if (hotwords == null)
                            stream.Hotwords = new List<string> { "魔搭", "开放时间" };
                        else
                            stream.Hotwords = hotwords.ToList();
                        if (languages.Length - 1 >= m)
                            stream.Language = languages[m];
                        else
                            stream.Language = languages.LastOrDefault();
                        stream.AddSamples(sample);
                        streams.Add(stream);
                        m++;
                    }
                    int n = 0;
                    List<OfflineRecognizerResultEntity> nativeResults = offlineRecognizer.GetResults(streams);
                    foreach (OfflineRecognizerResultEntity nativeResult in nativeResults)
                    {
                        var resultEntity = ConvertToResultEntity(nativeResult, n, (DateTime.Now - processStartTime).TotalMilliseconds / nativeResults.Count);
                        RaiseRecognitionResult(resultEntity);
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
            RaiseRecognitionCompleted(DateTime.Now - processStartTime, total_duration, samples.Count);
        }
        protected static AsrResultEntity ConvertToResultEntity(ASR.Model.OfflineRecognizerResultEntity nativeResult, int index, double processingTimeMs)
        {
            return new AsrResultEntity
            {
                Text = nativeResult.Text,
                Tokens = nativeResult.Tokens?.ToArray() ?? Array.Empty<string>(),
                Timestamps = nativeResult.Timestamps?.Select(ts => new[] { ts.First(), ts.Last() }).ToArray() ?? Array.Empty<int[]>(),
                Index = index,
                ProcessingTimeMs = processingTimeMs
            };
        }
    }
}
