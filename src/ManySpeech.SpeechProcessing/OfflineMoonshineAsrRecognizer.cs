using ManySpeech.SpeechProcessing.Base;
using ManySpeech.SpeechProcessing.Entities;
using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;

namespace ManySpeech.SpeechProcessing
{
    public partial class OfflineMoonshineAsrRecognizer : BaseAsr
    {
        private OfflineRecognizer? _recognizer;
        public OfflineRecognizer InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_recognizer == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string preprocessFilePath = modelBasePath + "./" + modelName + "/preprocess.int8.onnx";
                string encodeFilePath = modelBasePath + "./" + modelName + "/encode.int8.onnx";
                string cachedDecodeFilePath = modelBasePath + "./" + modelName + "/cached_decode.int8.onnx";
                string uncachedDecodeFilePath = modelBasePath + "./" + modelName + "/uncached_decode.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/conf.json";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
                //OfflineRecognizer offlineRecognizer = new OfflineRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, configFilePath: configFilePath, threadsNum: 1);
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

                    // Process preprocess path (priority: containing modelAccuracy>last one that matches prefix)
                    var preprocessCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("preprocess."))
                        .ToList();
                    if (preprocessCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredModel = preprocessCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        preprocessFilePath = preferredModel?.TargetPath ?? preprocessCandidates.Last().TargetPath;
                    }

                    // Process encode path (priority: containing modelAccuracy>last one that matches prefix)
                    var encodeCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("encode."))
                        .ToList();
                    if (encodeCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredModel = encodeCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        encodeFilePath = preferredModel?.TargetPath ?? encodeCandidates.Last().TargetPath;
                    }

                    // Process cachedDecode path
                    var cachedDecodeCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("cached_decode."))
                        .ToList();
                    if (cachedDecodeCandidates.Any())
                    {
                        var preferredModeleb = cachedDecodeCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        cachedDecodeFilePath = preferredModeleb?.TargetPath ?? cachedDecodeCandidates.Last().TargetPath;
                    }

                    // Process uncachedDecode path
                    var uncachedDecodeCandidates = fileInfos
                        .Where(f => f.FileName.StartsWith("uncached_decode."))
                        .ToList();
                    if (uncachedDecodeCandidates.Any())
                    {
                        var preferredModeleb = uncachedDecodeCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        uncachedDecodeFilePath = preferredModeleb?.TargetPath ?? uncachedDecodeCandidates.Last().TargetPath;
                    }
                    // Process token paths (take the last one that matches the prefix)
                    configFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("conf.") && f.FileName.EndsWith(".json"))
                        ?.TargetPath ?? "";

                    // Process token paths (take the last one that matches the prefix)
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("tokens.") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    if (new[] { preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, configFilePath, tokensFilePath }.Any(string.IsNullOrEmpty))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _recognizer = new OfflineRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
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
            return _recognizer;
        }

        public override async Task<List<AsrResultEntity>> RecognizeAsync(
            List<List<float[]>> samplesList,
            string modelBasePath,
            string modelName = "moonshine-tiny-en-onnx",
            string modelAccuracy = "int8",
            string streamDecodeMethod = "one",
            int threadsNum = 2)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            OfflineRecognizer? offlineRecognizer = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (offlineRecognizer == null)
            {
                throw new InvalidOperationException("Failed to initialize recognizer");
            }
            var results = new List<AsrResultEntity>();
            try
            {
                Console.WriteLine("Automatic speech recognition in progress!");
                DateTime processStartTime = DateTime.Now;
                streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "batch" : streamDecodeMethod;//one ,batch
                if (streamDecodeMethod == "one")
                {
                    // Non batch method
                    Console.WriteLine("Recognition results:\r\n");
                    try
                    {
                        for (int i = 0; i < samplesList.Count; i++)
                        {
                            OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                            foreach (var sample in samplesList[i])
                            {
                                stream.AddSamples(sample);
                            }
                            OfflineRecognizerResultEntity nativeResult = offlineRecognizer.GetResult(stream);
                            var processingTime = (DateTime.Now - processStartTime).TotalMilliseconds;
                            var resultEntity = ConvertToResultEntity(nativeResult, i, processingTime);
                            results.Add(resultEntity);
                            RaiseRecognitionResult(resultEntity);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        Console.WriteLine(ex.InnerException?.InnerException);
                    }
                    // Non batch method
                }
                if (streamDecodeMethod == "chunk")
                {
                    // Non batch method
                    Console.WriteLine("Recognition results:\r\n");
                    try
                    {
                        for (int i = 0; i < samplesList.Count; i++)
                        {
                            foreach (var sample in samplesList[i])
                            {
                                OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                                stream.AddSamples(sample);
                                OfflineRecognizerResultEntity nativeResult = offlineRecognizer.GetResult(stream);
                                var processingTime = (DateTime.Now - processStartTime).TotalMilliseconds;
                                var resultEntity = ConvertToResultEntity(nativeResult, i, processingTime);
                                results.Add(resultEntity);
                                RaiseRecognitionResult(resultEntity);
                            }

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
                        //int n = 0;
                        List<OfflineStream> streams = new List<OfflineStream>();
                        foreach (var sampleGroup in samplesList)
                        {
                            var stream = offlineRecognizer.CreateOfflineStream();
                            foreach (var sample in sampleGroup)
                            {
                                stream.AddSamples(sample);
                            }
                            streams.Add(stream);
                        }
                        var nativeResults = offlineRecognizer.GetResults(streams);
                        for (int i = 0; i < nativeResults.Count; i++)
                        {
                            var resultEntity = ConvertToResultEntity(nativeResults[i], i, (DateTime.Now - processStartTime).TotalMilliseconds / nativeResults.Count);
                            resultEntity.ModelName = modelName;
                            results.Add(resultEntity);
                            RaiseRecognitionResult(resultEntity);
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        Console.WriteLine(ex.InnerException?.InnerException.Message);
                    }
                    // batch method
                }
                int totalDurationMs = (int)samplesList.Select(x => x.Select(x => CalculateAudioDuration(x)).Sum()).Sum();
                RaiseRecognitionCompleted(DateTime.Now - processStartTime, TimeSpan.FromMilliseconds(totalDurationMs), samplesList.Count);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error occurred: {ex.Message}");
            }
            return results;
        }
        protected static AsrResultEntity ConvertToResultEntity(OfflineRecognizerResultEntity nativeResult, int index, double processingTimeMs)
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
        public override void Dispose()
        {
            if (!_disposed)
            {
                _recognizer?.Dispose();
                _recognizer = default;
                _disposed = true;
            }
        }
    }
}
