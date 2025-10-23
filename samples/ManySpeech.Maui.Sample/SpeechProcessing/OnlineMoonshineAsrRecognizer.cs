using ManySpeech.MoonshineAsr;
using ManySpeech.MoonshineAsr.Model;
using ManySpeech.Maui.Sample.SpeechProcessing.Base;
using ManySpeech.Maui.Sample.SpeechProcessing.Entities;
using System.Diagnostics;
using System.Text;

namespace ManySpeech.Maui.Sample.SpeechProcessing
{
    internal partial class OnlineMoonshineAsrRecognizer : BaseAsr
    {
        private string _lastResult = "";
        private string _lastResultPunc = "";
        private float[] _lastSample = new float[0];
        private List<int[]> _lastTimestamps = new List<int[]>() { new int[2] };
        private int _muteTimes;
        private StringBuilder _output = new StringBuilder();
        private int _i = 0;
        bool _isStart = false;
        private OnlineVadStream? _onlineStream = null;
        private DateTime _processStartTime;

        private OnlineVadRecognizer? _recognizer;
        public OnlineVadRecognizer? InitOnlineVadRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2, string vadModelName = "alifsmnvad-onnx")
        {
            if (_recognizer == null)
            {
                if (string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string preprocessFilePath = modelBasePath + "./" + modelName + "/preprocess.int8.onnx";
                string encodeFilePath = modelBasePath + "./" + modelName + "/encode.int8.onnx";
                string cachedDecodeFilePath = modelBasePath + "./" + modelName + "/cached_decode.int8.onnx";
                string uncachedDecodeFilePath = modelBasePath + "./" + modelName + "/uncached_decode.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/conf.json";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
                string vadModelFilePath = modelBasePath + "/" + vadModelName + "/" + "model.int8.onnx";
                string vadMvnFilePath = modelBasePath + vadModelName + "/" + "vad.mvn";
                string vadConfigFilePath = modelBasePath + vadModelName + "/" + "vad.json";
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
                    _recognizer = new OnlineVadRecognizer(preprocessFilePath, encodeFilePath, cachedDecodeFilePath, uncachedDecodeFilePath, tokensFilePath, vadModelFilePath: vadModelFilePath, vadConfigFilePath: vadConfigFilePath, vadMvnFilePath: vadMvnFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
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
                    Console.WriteLine($"Error occurred: {ex.Message}");
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
            OnlineVadRecognizer? onlineRecognizer = InitOnlineVadRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (onlineRecognizer == null)
            {
                throw new InvalidOperationException("Failed to initialize recognizer");
            }
            var results = new List<AsrResultEntity>();
            try
            {
                Console.WriteLine("Automatic speech recognition in progress!");
                var stopwatch = Stopwatch.StartNew();
                streamDecodeMethod = string.IsNullOrEmpty(streamDecodeMethod) ? "batch" : streamDecodeMethod;//one ,batch
                if (streamDecodeMethod == "one")
                {
                    //one stream decode
                    for (int j = 0; j < samplesList.Count; j++)
                    {
                        DateTime processStartTime = DateTime.Now;
                        using var stream = onlineRecognizer.CreateOnlineVadStream();
                        List<int[]> timestamps = new List<int[]>() { new int[2] };
                        foreach (float[] samplesItem in samplesList[j])
                        {
                            stream.AddSamples(samplesItem);
                            var nativeResult = onlineRecognizer.GetResult(stream);
                            var processingTime = (DateTime.Now - processStartTime).TotalMilliseconds;
                            var resultEntity = ConvertToResultEntity(nativeResult, j, processingTime);
                            resultEntity.ModelName = modelName;
                            if (samplesItem.SkipLast(Math.Min(samplesItem.Length, 1600)).Average() != 0)
                            {
                                timestamps.Add(new int[] { timestamps.Last()[1], timestamps.Last()[1] + (int)CalculateAudioDuration(samplesItem) });
                            }
                            resultEntity.Timestamps = timestamps.ToArray();
                            results.Add(resultEntity);
                            RaiseRecognitionResult(resultEntity);
                        }
                    }
                    // one stream decode
                }
                if (streamDecodeMethod == "batch")
                {
                    await ProcessMultiStream(onlineRecognizer, samplesList, modelName);
                }
                if (streamDecodeMethod == "chunk")
                {
                    for (int j = 0; j < samplesList.Count; j++)
                    {
                        var chunkResults = await ProcessChunkStream(onlineRecognizer, new List<List<float[]>> { samplesList[j] }, modelName);
                        results.AddRange(chunkResults);
                    }
                }
                stopwatch.Stop();
                int totalDurationMs = (int)samplesList.Select(x => x.Select(x => CalculateAudioDuration(x)).Sum()).Sum();
                RaiseRecognitionCompleted(stopwatch.Elapsed, TimeSpan.FromMilliseconds(totalDurationMs), results.Count);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error occurred: {ex.Message}");
            }
            return results;
        }

        private async Task<List<AsrResultEntity>> ProcessMultiStream(OnlineVadRecognizer recognizer, List<List<float[]>> samplesList, string modelName)
        {
            var results = new List<AsrResultEntity>();
            var onlineStreams = new List<OnlineVadStream>();
            var isEndpoints = new List<bool>();
            var isEnds = new List<bool>();
            List<List<int[]>> timestampsList = new List<List<int[]>>();
            try
            {
                // 初始化所有流
                for (int i = 0; i < samplesList.Count; i++)
                {
                    var stream = recognizer.CreateOnlineVadStream();
                    onlineStreams.Add(stream);
                    isEndpoints.Add(false);
                    isEnds.Add(false);
                    timestampsList.Add(new List<int[]> { new int[2] });

                    // 初始化结果对象
                    results.Add(new AsrResultEntity
                    {
                        Index = i,
                        ModelName = modelName,
                        Text = "",
                        Tokens = Array.Empty<string>(),
                        Timestamps = Array.Empty<int[]>()
                    });
                }

                int frameIndex = 0;
                bool hasMoreFrames = true;

                while (hasMoreFrames)
                {
                    var activeStreams = new List<OnlineVadStream>();
                    var streamIndices = new List<int>();

                    // 收集当前帧需要处理的流
                    for (int streamIndex = 0; streamIndex < samplesList.Count; streamIndex++)
                    {
                        if (isEnds[streamIndex]) continue;

                        if (frameIndex < samplesList[streamIndex].Count)
                        {
                            // 添加当前帧的样本
                            onlineStreams[streamIndex].AddSamples(samplesList[streamIndex][frameIndex]);
                            activeStreams.Add(onlineStreams[streamIndex]);
                            streamIndices.Add(streamIndex);
                            isEndpoints[streamIndex] = false;
                        }
                        else
                        {
                            // 标记流结束
                            isEndpoints[streamIndex] = true;
                            if (onlineStreams[streamIndex].IsFinished(true))
                            {
                                isEnds[streamIndex] = true;
                            }
                            else
                            {
                                activeStreams.Add(onlineStreams[streamIndex]);
                                streamIndices.Add(streamIndex);
                            }
                        }
                    }

                    if (activeStreams.Count > 0)
                    {
                        // 批量处理当前帧
                        var batchResults = recognizer.GetResults(activeStreams);

                        for (int resultIndex = 0; resultIndex < batchResults.Count; resultIndex++)
                        {
                            var streamIndex = streamIndices[resultIndex];
                            var nativeResult = batchResults[resultIndex];

                            var resultEntity = ConvertToResultEntity(nativeResult, streamIndex + 1, 0);
                            resultEntity.ModelName = modelName;
                            var samplesItem = samplesList[streamIndex][frameIndex];
                            if (samplesItem.SkipLast(Math.Min(samplesItem.Length, 1600)).Average() != 0)
                            {
                                timestampsList[streamIndex].Add(new int[] { timestampsList[streamIndex].Last()[1], timestampsList[streamIndex].Last()[1] + (int)CalculateAudioDuration(samplesItem) });
                            }
                            resultEntity.Timestamps = timestampsList[streamIndex].ToArray();
                            // 更新结果
                            results[streamIndex] = resultEntity;
                            RaiseRecognitionResult(resultEntity);
                        }

                        // 更新进度
                        var processedCount = results.Count(r => !string.IsNullOrEmpty(r.Text));
                    }

                    frameIndex++;

                    // 检查是否所有流都处理完成
                    hasMoreFrames = false;
                    for (int streamIndex = 0; streamIndex < samplesList.Count; streamIndex++)
                    {
                        if (!isEnds[streamIndex])
                        {
                            hasMoreFrames = true;
                            break;
                        }
                    }

                    // 添加小延迟以避免CPU过度使用
                    await Task.Delay(10);
                }

                // 最终处理所有流以确保获得最终结果
                var finalResults = recognizer.GetResults(onlineStreams);
                for (int i = 0; i < finalResults.Count; i++)
                {
                    var resultEntity = ConvertToResultEntity(finalResults[i], i + 1, 0);
                    resultEntity.ModelName = modelName;
                    resultEntity.Timestamps = timestampsList[i].ToArray();
                    results[i] = resultEntity;
                    RaiseRecognitionResult(resultEntity);
                }
            }
            finally
            {
                // 清理所有流
                foreach (var stream in onlineStreams)
                {
                    stream.Dispose();
                }
            }
            return results;
        }

        private async Task<List<AsrResultEntity>> ProcessChunkStream(OnlineVadRecognizer recognizer, List<List<float[]>> samplesList, string modelName)
        {
            var results = new List<AsrResultEntity>();
            int i = 0;
            if (_onlineStream == null)
            {
                _onlineStream = recognizer.CreateOnlineVadStream();
                _processStartTime = DateTime.Now;
            }
            float[] sample = samplesList[i][0];
            Array.Resize(ref _lastSample, _lastSample.Length + sample.Length);
            Array.Copy(sample, 0, _lastSample, _lastSample.Length - sample.Length, sample.Length);
            int maxMuteTimes = 1;
            int minSentenceLength = 8;
            if (!_isStart)
            {
                _isStart = true;
            }
            _onlineStream.AddSamples(sample);
            if (_muteTimes == maxMuteTimes)
            {
                _onlineStream.AddSamples(new float[2400]);
                _onlineStream.AddSamples(new float[2400]);
            }
            var nativeResult = recognizer.GetResult(_onlineStream);
            var processingTime = (DateTime.Now - _processStartTime).TotalMilliseconds;
            bool sentenceIfNeed = true;
            if (_muteTimes <= maxMuteTimes)
            {
                _lastTimestamps.Add(new int[] { _lastTimestamps.Last()[1], _lastTimestamps.Last()[1] + (int)CalculateAudioDuration(sample) });
                if (nativeResult != null && nativeResult.Text?.Trim().Length > 0)
                {
                    if (nativeResult.Text.CompareTo(_lastResult) == 0)
                    {
                        _muteTimes++;
                    }
                    else
                    {
                        _muteTimes = 0;
                    }
                    _lastResult = nativeResult.Text;
                    _lastResultPunc = _lastResult;
                    if (sentenceIfNeed) _output = new StringBuilder();
                    nativeResult.Text = string.Format("{0}{2}", _output.ToString(), _i, _lastResultPunc);
                    var resultEntity = ConvertToResultEntity(nativeResult, _i, processingTime);
                    resultEntity.ModelName = modelName;
                    resultEntity.Timestamps = _lastTimestamps.ToArray();//置换为最新时间戳
                    RaiseRecognitionResult(resultEntity);
                    results.Clear();
                    results.Add(resultEntity);
                }
            }
            if (_muteTimes > maxMuteTimes || (nativeResult != null && nativeResult.Text?.Length > 0))
            {
                _output.Append(string.Format("{1}", _i, _lastResultPunc));
                nativeResult.Text = string.Format("{0}", _output.ToString());
                var resultEntity = ConvertToResultEntity(nativeResult, _i, processingTime);
                resultEntity.ModelName = modelName;
                resultEntity.Timestamps = _lastTimestamps.ToArray();
                RaiseRecognitionResult(resultEntity);
                RaiseRecognitionCompleted(TimeSpan.FromMilliseconds(processingTime), TimeSpan.FromMilliseconds(CalculateAudioDuration(_lastSample)), results.Count, _lastSample);
                if (sentenceIfNeed) _output = new StringBuilder();
                results.Clear();
                results.Add(resultEntity);
                _onlineStream = recognizer.CreateOnlineVadStream();
                _processStartTime = DateTime.Now;
                _lastResult = "";
                _lastSample = new float[0];
                _lastTimestamps = new List<int[]>() { new int[] { _lastTimestamps.Last()[1], _lastTimestamps.Last()[1] } };
                _muteTimes = 0;
                _i++;
                _onlineStream.AddSamples(new float[2400]);
                _onlineStream.AddSamples(new float[2400]);

            }
            return results;
        }
        protected static AsrResultEntity ConvertToResultEntity(OnlineRecognizerResultEntity nativeResult, int index, double processingTimeMs)
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
