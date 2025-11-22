using ManySpeech.AliParaformerAsr;
using ManySpeech.AliParaformerAsr.Model;
using ManySpeech.SpeechProcessing.Base;
using ManySpeech.SpeechProcessing.Entities;

namespace ManySpeech.SpeechProcessing
{
    public partial class OfflineAliParaformerAsrRecognizer : BaseAsr
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
                string modelFilePath = modelBasePath + "/" + modelName + "/model.int8.onnx";
                string configFilePath = modelBasePath + "/" + modelName + "/asr.yaml";
                string mvnFilePath = modelBasePath + "/" + modelName + "/am.mvn";
                string tokensFilePath = modelBasePath + "/" + modelName + "/tokens.txt";
                string modelebFilePath = modelBasePath + "/" + modelName + "/model_eb.int8.onnx";
                string hotwordFilePath = modelBasePath + "/" + modelName + "/hotword.txt";
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
                    _recognizer = new OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath, threadsNum: threadsNum);
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
            string modelName = "paraformer-seaco-large-zh-timestamp-onnx-offline",
            string modelAccuracy = "int8",
            string streamDecodeMethod = "one",
            int threadsNum = 2)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            OfflineRecognizer offlineRecognizer = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
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
                            // Modify the logic here to dynamically modify hot words
                            //stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(modelBasePath, modelName, "tokens.txt"), new string[] {"魔搭" }); 
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
                                // Modify the logic here to dynamically modify hot words
                                //stream.Hotwords = Utils.TextHelper.GetHotwords(Path.Combine(modelBasePath, modelName, "tokens.txt"), new string[] {"魔搭" }); 
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
        protected static AsrResultEntity ConvertToResultEntity(AliParaformerAsr.Model.OfflineRecognizerResultEntity nativeResult, int index, double processingTimeMs)
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
