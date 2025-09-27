using System.Text;
using ManySpeech.AliCTTransformerPunc;

namespace ManySpeech.AliCTTransformerPunc.Examples
{
    internal static partial class Program
    {
        private static CTTransformer? _cttPunc;
        public static CTTransformer InitOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_cttPunc == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string modelFilePath = modelBasePath + "./" + modelName + "/model.int8.onnx";
                string configFilePath = modelBasePath + "./" + modelName + "/punc.json";
                string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
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
                        .LastOrDefault(f => f.FileName.StartsWith("punc") && (f.FileName.EndsWith(".json")))
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
                    _cttPunc = new CTTransformer(modelFilePath: modelFilePath, configFilePath: configFilePath, tokensFilePath: tokensFilePath, threadsNum: threadsNum);
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
            return _cttPunc;
        }
        public static void AutoPunctuationWithFile(string modelName = "alicttransformerpunc-large-zh-en-int8-onnx", string modelAccuracy = "int8", int threadsNum = 2, string[]? filePaths = null, string? modelBasePath = null, int splitSize = 15)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            CTTransformer cttPunc = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (cttPunc == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            int totalWordsNum = 0;
            List<string[]>? textList = new List<string[]>();
            List<string> paths = new List<string>();
            if (filePaths == null || filePaths.Count() == 0)
            {
                filePaths = Directory.GetFiles(Path.Join(modelBasePath, modelName, "example"));
            }
            foreach (string filePath in filePaths)
            {
                if (!File.Exists(filePath))
                {
                    continue;
                }
                if (PreProcessUtils.TextHelper.IsTextByHeader(filePath))
                {
                    int wordsNum = 0;
                    string[]? texts = PreProcessUtils.TextHelper.GetFileText(filePath: filePath, wordsNum: ref wordsNum);
                    if (texts != null)
                    {
                        paths.Add(filePath);
                        textList.Add(texts);
                        totalWordsNum += wordsNum;
                    }
                }
            }
            if (textList.Count == 0)
            {
                Console.WriteLine("No text file is read!");
                return;
            }
            Console.WriteLine("Automatic Punctuation recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            int n = 0;
            foreach (string[] texts in textList)
            {
                Console.WriteLine($"{paths[n]}");
                foreach (string text in texts)
                {
                    string result = cttPunc.GetResults(text, splitSize: splitSize);
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
                n++;
            }
            if (_cttPunc != null)
            {
                _cttPunc.Dispose();
                _cttPunc = null;
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = totalWordsNum / elapsed_milliseconds * 1000;
            Console.WriteLine("punc_elapsed_seconds:{0}", ((double)(elapsed_milliseconds / 1000)).ToString());
            Console.WriteLine("total_words_number:{0}", totalWordsNum.ToString());
            Console.WriteLine("words_per_second:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }

        public static void AutoPunctuationWithText(string modelName = "alicttransformerpunc-large-zh-en-int8-onnx", string modelAccuracy = "int8", int threadsNum = 2, string? str = null, string? modelBasePath = null, int splitSize = 15)
        {
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            CTTransformer cttPunc = InitOfflineRecognizer(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (cttPunc == null)
            {
                Console.WriteLine("Init models failure!");
                return;
            }
            int totalWordsNum = 0;
            
            if (string.IsNullOrEmpty(str))
            {
                Console.WriteLine("No text content is read!");
                return;
            }
            int wordsNum = 0;
            string[]? texts = PreProcessUtils.TextHelper.GetStrText(str: str, wordsNum: ref wordsNum);
            totalWordsNum += wordsNum;            
            Console.WriteLine("Automatic Punctuation recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);            
            foreach (string text in texts)
            {
                string result = cttPunc.GetResults(text, splitSize: splitSize);
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
            if (_cttPunc != null)
            {
                _cttPunc.Dispose();
                _cttPunc = null;
            }
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = totalWordsNum / elapsed_milliseconds * 1000;
            Console.WriteLine("punc_elapsed_seconds:{0}", ((double)(elapsed_milliseconds / 1000)).ToString());
            Console.WriteLine("total_words_number:{0}", totalWordsNum.ToString());
            Console.WriteLine("words_per_second:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("end!");
        }
    }
}
