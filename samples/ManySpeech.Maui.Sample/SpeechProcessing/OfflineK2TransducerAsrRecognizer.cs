using ManySpeech.Maui.Sample.SpeechProcessing.Base;
using ManySpeech.Maui.Sample.SpeechProcessing.Entities;
using ManySpeech.K2TransducerAsr;
using ManySpeech.K2TransducerAsr.Model;
using PreProcessUtils;

namespace ManySpeech.Maui.Sample.SpeechProcessing
{
    internal partial class OfflineK2TransducerAsrRecognizer : BaseAsr
    {
        private OfflineRecognizer? _recognizer;
        private OfflineRecognizer? InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_recognizer == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string encoderFilePath = modelBasePath + "/" + modelName + "/model.int8.onnx";
                string decoderFilePath = "";
                string joinerFilePath = "";
                string tokensFilePath = modelBasePath + "/" + modelName + "/tokens.txt";
                try
                {
                    string folderPath = Path.Combine(modelBasePath, modelName);
                    // 1. Check if the folder exists
                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine($"Error: folder does not exist - {folderPath}");
                        return null;
                    }
                    // 2. Get the complete paths of all files in the folder
                    // Optional parameters: search mode (such as "*. txt" filtering text files), whether to search subdirectories
                    string[] allFilePaths = Directory.GetFiles(folderPath);
                    foreach (string filePath in allFilePaths)
                    {
                        // Extract pure file name (including extension)
                        string fileName = Path.GetFileName(filePath);
                        //Console.WriteLine(fileName);
                        if (fileName.StartsWith("model") || fileName.StartsWith("encoder"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                encoderFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(encoderFilePath))
                                {
                                    encoderFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("decoder"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                decoderFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(decoderFilePath))
                                {
                                    decoderFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("joiner"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                joinerFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(joinerFilePath))
                                {
                                    joinerFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("tokens"))
                        {
                            tokensFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                        }
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _recognizer = new OfflineRecognizer(encoderFilePath, decoderFilePath, joinerFilePath, tokensFilePath, threadsNum: threadsNum);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
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
                    Console.WriteLine($"Error occurred: {ex.Message}");
                }
            }
            return _recognizer;
        }

        public override async Task<List<AsrResultEntity>> RecognizeAsync(
            List<List<float[]>> samplesList,
            string modelBasePath,
            string modelName = "k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716",
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
