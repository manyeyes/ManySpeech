// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
/*
 * Before running, please prepare the model first
 * Model Download:
 * Please read README.md
 */
using ManySpeech.K2TransducerAsr.Examples;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.k2TransducerAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        private static GitHelper _modelPreparer = new GitHelper();
        // language
        private static string _lang = "en";
        // environment variable prefix (to avoid naming conflicts)
        private const string EnvPrefix = "MANYSPEECH_";
        // supported environment variables
        private const string EnvModelBasePath = EnvPrefix + "BASE";    // path/to/directory
        private const string EnvRecognizerType = EnvPrefix + "TYPE";    // online/offline
        private const string EnvBatchType = EnvPrefix + "BATCH";       // one/batch
        private const string EnvModelName = EnvPrefix + "MODEL";       // model name
        private const string EnvModelAccuracy = EnvPrefix + "ACCURACY";       // model accuracy int8/fp32
        private const string EnvThreads = EnvPrefix + "THREADS";       // thread num
                                                                       // The complete model path, eg: path/to/directory/modelname
        private const string _modelBasePath = @"";// eg: path/to/directory. It is the root directory where the model is stored. If it is empty, the program root directory will be read by default.
                                                  // default-model
        private static Dictionary<string, string> _defaultOnlineModelName = new Dictionary<string, string>{
            { "k2transducerasr", "k2transducer-zipformer-ctc-small-zh-onnx-online-20250401" }
        };
        private static Dictionary<string, string> _defaultOfflineModelName = new Dictionary<string, string>{
            { "k2transducerasr", "k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716" } };

        [STAThread]
        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            Console.WriteLine($"Choose the language for Usage: 1. English; 2. Chinese;");
            int selectLanguage = 0;
            if (int.TryParse(GetConsoleReadLine().Trim(), out selectLanguage))
            {
                _lang = selectLanguage == 2 ? "zh" : "en";
            }
            PrintUsage();

            while (true)
            {
                try
                {
                    if (args.Length == 0)
                    {
                        // 1. use Environment. GetCommandLineArgs() to obtain complete command-line information
                        args = Environment.GetCommandLineArgs();
                        if (args.Length == 1)
                        {
                            Console.WriteLine("\nEnter parameters (press Enter to skip):");
                        }
                        args = ParseArguments(GetConsoleReadLine().Trim());
                    }

                    string[] allArgs = args;

                    // extract actual parameters (if the first element does not start with a "-", exclude: program path)
                    string[] commandLineArgs = allArgs.Length > 1
                        ? !allArgs[0].StartsWith("-") ? allArgs[1..] : allArgs
                        : Array.Empty<string>();

                    if (commandLineArgs.Length == 0)
                    {
                        Console.WriteLine($"Select example: 1.offline; 2.online;");
                        int selectExample = 0;
                        if (int.TryParse(GetConsoleReadLine().Trim(), out selectExample))
                        {
                            switch (selectExample)
                            {
                                case 1:
                                    commandLineArgs = new string[] { "-type", "offline" };
                                    break;
                                case 2:
                                    commandLineArgs = new string[] { "-type", "online" };
                                    break;
                            }
                        }
                    }

                    // 2. read environment variables as default values
                    var envConfig = new Dictionary<string, string?>
                {
                    { "modelBasePath", Environment.GetEnvironmentVariable(EnvModelBasePath)??""},
                    { "recognizerType", Environment.GetEnvironmentVariable(EnvRecognizerType)},
                    { "methodType", Environment.GetEnvironmentVariable(EnvBatchType) ?? "one" },
                    { "modelName", Environment.GetEnvironmentVariable(EnvModelName) ?? "default-model" },
                    { "modelAccuracy", Environment.GetEnvironmentVariable(EnvModelAccuracy) ?? "int8" },
                    { "threads", Environment.GetEnvironmentVariable(EnvThreads) ?? "2" }
                };

                    // 3. resolve command-line parameters (overwrite environment variables)
                    var appConfig = ParseArgs(commandLineArgs, envConfig);

                    // 4. execute the corresponding recognizer method
                    ExecuteRecognizer(appConfig);
                }
                catch (ArgumentException ex)
                {
                    Console.WriteLine($"parameter error: {ex.Message}");
                    PrintUsage();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"program error: {ex.Message}");
                }
                args = Array.Empty<string>();
            }
        }
        private static string GetConsoleReadLine()
        {
            return Console.ReadLine()?.Trim('\r', '\n') ?? string.Empty;
        }

        public static string[] ParseArguments(string input)
        {
            List<string> args = new List<string>();
            string currentArg = "";
            bool inQuotes = false;
            char quoteChar = '\0';

            // 替换换行符并处理输入
            input = input.Replace("\r\n", " ");

            for (int i = 0; i < input.Length; i++)
            {
                char c = input[i];

                // 处理引号
                if (c == '"' || c == '\'')
                {
                    // 如果是相同的引号，则认为是结束
                    if (inQuotes && c == quoteChar)
                    {
                        inQuotes = false;
                        quoteChar = '\0';
                    }
                    // 如果是新的引号，则开始
                    else if (!inQuotes)
                    {
                        inQuotes = true;
                        quoteChar = c;
                    }
                    // 如果是不同的引号且在引号内，则视为普通字符
                    else
                    {
                        currentArg += c;
                    }
                }
                // 处理空格
                else if (char.IsWhiteSpace(c))
                {
                    // 如果在引号内，空格是参数的一部分
                    if (inQuotes)
                    {
                        currentArg += c;
                    }
                    // 如果不在引号内，空格是参数分隔符
                    else
                    {
                        if (!string.IsNullOrEmpty(currentArg))
                        {
                            args.Add(currentArg);
                            currentArg = "";
                        }
                    }
                }
                // 普通字符
                else
                {
                    currentArg += c;
                }
            }

            // 添加最后一个参数
            if (!string.IsNullOrEmpty(currentArg))
            {
                args.Add(currentArg);
            }

            return args.ToArray();
        }

        /// <summary>
        /// analyze command-line parameters and merge default values of environment variables
        /// </summary>
        private static Dictionary<string, object> ParseArgs(string[] args, Dictionary<string, string?> envConfig)
        {
            var config = new Dictionary<string, object>
            {
                // Initialize default values (from environment variables)
                { "modelBasePath", envConfig["modelBasePath"] },
                { "recognizerType", envConfig["recognizerType"] },
                { "methodType", envConfig["methodType"] ?? "one" },
                { "modelName", envConfig["modelName"] },
                { "modelAccuracy", envConfig["modelAccuracy"] },
                { "threads", int.Parse(envConfig["threads"]!) },
                { "files", Array.Empty<string>() }
            };

            // resolve command-line parameters (override default values)
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i].ToLower())
                {
                    case "-base":
                        if (i + 1 < args.Length)
                            config["modelBasePath"] = args[++i];
                        break;
                    case "-type":
                        if (i + 1 < args.Length)
                            config["recognizerType"] = args[++i];
                        break;
                    case "-method":
                        if (i + 1 < args.Length)
                            config["methodType"] = args[++i];
                        break;
                    case "-model":
                        if (i + 1 < args.Length)
                            config["modelName"] = args[++i];
                        break;
                    case "-accuracy":
                        if (i + 1 < args.Length)
                            config["modelAccuracy"] = args[++i];
                        break;
                    case "-threads":
                        if (i + 1 < args.Length && int.TryParse(args[++i], out int threads))
                            config["threads"] = threads;
                        else
                            throw new ArgumentException("The number of threads must be a valid integer");
                        break;
                    case "-files":
                        var files = new List<string>();
                        while (++i < args.Length && !args[i].StartsWith("-"))
                            files.Add(args[i].Trim('\"'));
                        config["files"] = files.ToArray();
                        i--; // fix index position
                        break;
                    default:
                        throw new ArgumentException($"Unknown parameters: {args[i]}");
                }
            }

            // verify required parameters：-type is mandatory
            if (config["recognizerType"] == null)
                throw new ArgumentException("You must specify the recognizer type (-type online/offline) or set an environment variable " + EnvRecognizerType);

            return config;
        }

        /// <summary>
        /// execute the corresponding recognizer method
        /// </summary>
        private static async void ExecuteRecognizer(Dictionary<string, object> config)
        {
            string modelBasePath = config["modelBasePath"].ToString()!.ToLower();
            string recognizerType = config["recognizerType"].ToString()!.ToLower();
            string methodType = config["methodType"].ToString()!.ToLower();
            string modelName = config["modelName"].ToString()!;
            string modelAccuracy = config["modelAccuracy"].ToString()!;
            int threads = (int)config["threads"];
            string[] files = (string[])config["files"];
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }

            string defaultOnlineModelName = _defaultOnlineModelName.GetValueOrDefault("k2transducerasr");
            string defaultOfflineModelName = _defaultOfflineModelName.GetValueOrDefault("k2transducerasr");
            modelName = modelName == "default-model" ? (recognizerType == "online" ? defaultOnlineModelName : defaultOfflineModelName) : modelName;
            if (!string.IsNullOrEmpty(modelName))
                await Task.Run(() => _modelPreparer.ProcessCloneModel(modelBasePath, modelName));
            PrintConfigInfo(modelBasePath, recognizerType, methodType, modelName, modelAccuracy,
                           threads, files);
            try
            {
                // 调用对应的识别方法
                if (recognizerType == "online")
                {
                    if (modelName == "default-model")
                    {
                        modelName = defaultOnlineModelName;
                    }
                    SetOnlineRecognizerCallbackForResult(recognizerType: recognizerType);
                    SetOnlineRecognizerCallbackForCompleted();
                    OnlineK2TransducerAsrRecognizer.OnlineRecognizer(methodType, modelName, modelAccuracy, threads, files, modelBasePath);
                }
                else if (recognizerType == "offline")
                {
                    if (modelName == "default-model")
                    {
                        modelName = defaultOfflineModelName;
                    }
                    SetOfflineRecognizerCallbackForResult(recognizerType: recognizerType);
                    SetOfflineRecognizerCallbackForCompleted();
                    OfflineK2TransducerAsrRecognizer.OfflineRecognizer(methodType, modelName, modelAccuracy, threads, files, modelBasePath);
                }
                else
                {
                    throw new ArgumentException("The recognizer type must be online or offline");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex);
            }
            finally
            {
                //Suggest GC recycling (non mandatory)
                GC.Collect(); //Trigger recycling
                GC.WaitForPendingFinalizers(); //Waiting for the terminator to complete execution (such as Dispose logic)
                GC.Collect(); //Recycling again (ensuring that the resources released by the terminator are recycled)
            }
        }

        #region callback
        private static async void SetOfflineRecognizerCallbackForResult(string? recognizerType, string outputFormat = "text")
        {
            int i = 0;
            OfflineK2TransducerAsrRecognizer.ResetRecognitionResultHandlers();
            OfflineK2TransducerAsrRecognizer.OnRecognitionResult += async result =>
            {
                string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
                if (!string.IsNullOrEmpty(text))
                {
                    int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                    switch (outputFormat)
                    {
                        case "text":
                            Console.WriteLine($"[{recognizerType} Stream {resultIndex}]");
                            Console.WriteLine(text);
                            break;
                        case "json":
                            Console.WriteLine($"[{recognizerType} Stream {resultIndex}]");
                            StringBuilder r = new StringBuilder();
                            r.AppendLine("{");
                            r.AppendLine($"\"text\": \"{text}\",");
                            if (result.Tokens.Length > 0)
                            {
                                r.AppendLine($"\"tokens\":[{string.Join(",", result.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                            }
                            if (result.Timestamps.Length > 0)
                            {
                                r.AppendLine($"\"timestamps\":[{string.Join(",", result.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                            }
                            r.AppendLine("}");
                            Console.WriteLine($"{r.ToString()}");
                            break;
                    }
                }
                i++;
            };
        }
        private static async void SetOnlineRecognizerCallbackForResult(string? recognizerType, string outputFormat = "text")
        {
            int i = 0;
            OnlineK2TransducerAsrRecognizer.ResetRecognitionResultHandlers();
            OnlineK2TransducerAsrRecognizer.OnRecognitionResult += async result =>
            {
                string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
                if (!string.IsNullOrEmpty(text))
                {
                    int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                    switch (outputFormat)
                    {
                        case "text":
                            Console.WriteLine($"[{recognizerType} Stream {resultIndex}]");
                            Console.WriteLine(text);
                            break;
                        case "json":
                            Console.WriteLine($"[{recognizerType} Stream {resultIndex}]");
                            StringBuilder r = new StringBuilder();
                            r.AppendLine("{");
                            r.AppendLine($"\"text\": \"{text}\",");
                            if (result.Tokens.Length > 0)
                            {
                                r.AppendLine($"\"tokens\":[{string.Join(",", result.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                            }
                            if (result.Timestamps.Length > 0)
                            {
                                r.AppendLine($"\"timestamps\":[{string.Join(",", result.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                            }
                            r.AppendLine("}");
                            Console.WriteLine($"{r.ToString()}");
                            break;
                    }
                }
                i++;
            };
        }
        public static void SetOnlineRecognizerCallbackForCompleted()
        {
            OnlineK2TransducerAsrRecognizer.ResetRecognitionCompletedHandlers();
            OnlineK2TransducerAsrRecognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
            {
                double elapsedMilliseconds = totalTime.TotalMilliseconds;
                Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
                Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
                Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
            };
        }
        public static void SetOfflineRecognizerCallbackForCompleted()
        {
            OfflineK2TransducerAsrRecognizer.ResetRecognitionCompletedHandlers();
            OfflineK2TransducerAsrRecognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
            {
                double elapsedMilliseconds = totalTime.TotalMilliseconds;
                Console.WriteLine("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString());
                Console.WriteLine("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString());
                Console.WriteLine("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString());
            };
        }
        #endregion

        #region print
        private static void PrintConfigInfo(string modelBasePath, string recognizerType, string methodType,
                                         string modelName, string modelAccuracy, int threads, string[] files)
        {
            if (_lang.ToLower() == "zh")
            {
                Console.WriteLine($"===== 识别器配置 =====");
                Console.WriteLine($"模型目录: {modelBasePath ?? ""}");
                Console.WriteLine($"类型: {recognizerType}");
                Console.WriteLine($"批量模式: {methodType}（默认: one）");
                Console.WriteLine(string.Format("模型: {0}", modelName));
                Console.WriteLine($"精度: {modelAccuracy}");
                Console.WriteLine($"线程数: {threads}");
                Console.WriteLine($"输入文件: {(files.Length > 0 ? string.Join(", ", files) : "无")}");
                Console.WriteLine("======================");
            }
            else
            {
                Console.WriteLine("===== RecognizerConfiguration =====");
                Console.WriteLine($"Model Directory: {modelBasePath ?? ""}");
                Console.WriteLine($"Type:{recognizerType}");
                Console.WriteLine($"Batch Mode: {methodType} (default: one)");
                Console.WriteLine(string.Format("Model: {0}", modelName));
                Console.WriteLine($"Precision:{modelAccuracy}");
                Console.WriteLine($"Number of Threads: {threads}");
                Console.WriteLine($"Input Files: {(files.Length > 0 ? string.Join(", ", files) : "None")}");
                Console.WriteLine("==================================");
            }
        }

        /// <summary>
        /// 打印命令行使用说明
        /// </summary>
        private static void PrintUsage()
        {
            if (_lang.ToLower() == "zh")
            {
                Console.WriteLine("\n使用说明: k2TransducerAsr.Examples.exe [参数]");
                Console.WriteLine("必选参数（或通过环境变量设置）:");
                Console.WriteLine($"  -type <online/offline>   识别器类型（环境变量: {EnvRecognizerType}）");
                Console.WriteLine("可选参数:");
                Console.WriteLine($"  -method <one/batch>       批量处理模式（默认: one，环境变量: {EnvBatchType}）");
                Console.WriteLine($"  -base <可指定模型存放目录，或为空>   模型存放目录（环境变量: {EnvModelBasePath}）");
                Console.WriteLine($"  -model <名称>            模型名称（默认: default-model，环境变量: {EnvModelName}）");
                Console.WriteLine($"  -accuracy <fp32/int8>    模型名称（默认: int8，环境变量: {EnvModelAccuracy}）");
                Console.WriteLine($"  -threads <数量>          线程数（默认: 2，环境变量: {EnvThreads}）");
                Console.WriteLine("  -files <文件1> <文件2>    输入媒体文件列表(如不指定，默认:自动检查并识别模型目录下test_wavs中的文件)");
                Console.WriteLine("\n示例1:");
                Console.WriteLine("  k2TransducerAsr.Examples.exe -type online -method one -base /path/to/directory -model my-model -accuracy int8 -threads 2 -files /path/to/0.wav /path/to/1.wav");
                Console.WriteLine("\n示例2（使用默认method=one）:");
                Console.WriteLine($"  set {EnvRecognizerType}=online && set {EnvModelBasePath}=/path/to/directory && k2TransducerAsr.Examples.exe");
                Console.WriteLine($"\n*应用程序目录：{applicationBase}, 如果不指定-base, 请将下载的模型存放于此目录。");
                Console.WriteLine($"\n*附加说明：按2次回车，可根据提示操作：1.选择语言；2.选择识别器类型。");
            }
            else
            {
                Console.WriteLine("\nUsage Instructions: k2TransducerAsr.Examples.exe [parameters]");
                Console.WriteLine("Required parameters (or set via environment variables):");
                Console.WriteLine($"  -type <online/offline>   Recognizer type (environment variable: {EnvRecognizerType})");
                Console.WriteLine("Optional parameters:");
                Console.WriteLine($"  -method <one/batch>       Batch processing mode (default: one, environment variable: {EnvBatchType})");
                Console.WriteLine($"  -base <specifiable model directory, or empty>   Model storage directory (environment variable: {EnvModelBasePath})");
                Console.WriteLine($"  -model <name>            Model name (default: default-model, environment variable: {EnvModelName})");
                Console.WriteLine($"  -accuracy <fp32/int8>    Precision (default: int8, environment variable: {EnvModelAccuracy})");
                Console.WriteLine($"  -threads <count>         Number of threads (default: 2, environment variable: {EnvThreads})");
                Console.WriteLine("  -files <file1> <file2>    List of input media files (if not specified, default: automatically check and recognize files in test_wavs under the model directory)");
                Console.WriteLine("\nExample 1:");
                Console.WriteLine("  k2TransducerAsr.Examples.exe -type online -method one -base /path/to/directory -model my-model -accuracy int8 -threads 2 -files /path/to/0.wav /path/to/1.wav");
                Console.WriteLine("\nExample 2 (use default method=one):");
                Console.WriteLine($"  set {EnvRecognizerType}=online && set {EnvModelBasePath}=/path/to/directory && k2TransducerAsr.Examples.exe");
                Console.WriteLine($"\n*Application directory: {applicationBase}. If -base is not specified, please place the downloaded model in this directory.");
                Console.WriteLine($"\n*Additional notes: Press Enter twice, and you can follow the prompts to proceed: 1. Select language; 2. Select recognizer type.");
            }
        }
        #endregion
    }
}