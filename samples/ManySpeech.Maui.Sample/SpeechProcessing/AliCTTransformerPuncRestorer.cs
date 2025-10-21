using ManySpeech.AliCTTransformerPunc;
using System.Text;

namespace ManySpeech.Maui.Sample.SpeechProcessing
{
    internal partial class AliCTTransformerPuncRestorer : IDisposable
    {
        public bool _disposed = false;

        private string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        private CTTransformer? _cttpuncRestorer;
        public CTTransformer InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_cttpuncRestorer == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string modelFilePath = modelBasePath + "/" + modelName + "/model.int8.onnx";
                string configFilePath = modelBasePath + "/" + modelName + "/punc.json";
                string tokensFilePath = modelBasePath + "/" + modelName + "/tokens.txt";
                try
                {
                    string folderPath = Path.Join(modelBasePath, modelName);
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
                        .Where(f => f.FileName.StartsWith("model") && !f.FileName.Contains(".mge"))
                        .ToList();
                    if (modelCandidates.Any())
                    {
                        // Prioritize selecting files that contain the specified model accuracy
                        var preferredModel = modelCandidates
                            .LastOrDefault(f => f.FileName.Contains($".{modelAccuracy}."));
                        modelFilePath = preferredModel?.TargetPath ?? modelCandidates.Last().TargetPath;
                    }

                    // Process config paths (take the last one that matches the prefix)
                    configFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("punc") && f.FileName.EndsWith(".json"))
                        ?.TargetPath ?? "";

                    // Process token paths (take the last one that matches the prefix)
                    tokensFilePath = fileInfos
                        .LastOrDefault(f => f.FileName.StartsWith("tokens") && f.FileName.EndsWith(".txt"))
                        ?.TargetPath ?? "";

                    if (string.IsNullOrEmpty(modelFilePath) || string.IsNullOrEmpty(tokensFilePath))
                    {
                        return null;
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _cttpuncRestorer = new CTTransformer(modelFilePath: modelFilePath, configFilePath: configFilePath, tokensFilePath: tokensFilePath, threadsNum: threadsNum);
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
            return _cttpuncRestorer;
        }

        public void AutoPunctuationWithText(string[]? texts, string? modelBasePath, string modelName = "alicttransformerpunc-large-zh-en-int8-onnx", string modelAccuracy = "int8", int threadsNum = 2, int splitSize = 15)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            CTTransformer cttpuncRestorer = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (cttpuncRestorer == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            int totalWordsNum = 0;
            if (texts == null || texts.Length == 0)
            {
                Console.WriteLine("No text content is read!");
                return;
            }
            int wordsNum = 0;
            totalWordsNum += wordsNum;
            Console.WriteLine("Automatic Punctuation recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            foreach (string text in texts)
            {
                if (string.IsNullOrEmpty(text))
                {
                    continue;
                }
                string result = cttpuncRestorer.GetResults(text, splitSize: splitSize);
                if (result != null)
                {
                    StringBuilder r = new StringBuilder();
                    r.Append("{");
                    r.Append($"\"text\": \"{result}\"");
                    r.Append("}");
                    Console.WriteLine(r.ToString());
                    Console.WriteLine();
                }
            }
            if (_cttpuncRestorer != null)
            {
                _cttpuncRestorer.Dispose();
                _cttpuncRestorer = null;
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = totalWordsNum / elapsed_milliseconds * 1000;
            Console.WriteLine("punc_elapsed_seconds:{0}", ((double)(elapsed_milliseconds / 1000)).ToString());
            Console.WriteLine("total_words_number:{0}", totalWordsNum.ToString());
            Console.WriteLine("words_per_second:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }
        public void Dispose()
        {
            if (!_disposed)
            {
                _cttpuncRestorer?.Dispose();
                _cttpuncRestorer = default;
                _disposed = true;
            }
        }
    }
}
