// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
/*
 * Before running, please prepare the model first
 * Model Download:
 * Please read README.md
 */
namespace ManySpeech.AliCTTransformerPunc.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        private static string _lang = "en";
        private const string EnvPrefix = "MANYSPEECH_";
        private const string EnvModelBasePath = EnvPrefix + "BASE";
        private const string EnvModelName = EnvPrefix + "MODEL";
        private const string EnvModelAccuracy = EnvPrefix + "ACCURACY";
        private const string EnvThreads = EnvPrefix + "THREADS";
        private const string _modelBasePath = @"";
        private static Dictionary<string, string> _defaultModelName = new Dictionary<string, string>{
            { "alicttransformer", "alicttransformerpunc-large-zh-en-int8-onnx" } };
        private static int i = 0;

        [STAThread]
        static void Main(string[] args)
        {
            while (true)
            {
                try
                {
                    if (args.Length == 0)
                    {
                        args = Environment.GetCommandLineArgs();
                        if (args.Length == 1)
                        {
                            if (i == 0)
                                PrintUsage();
                            i++;
                            Console.WriteLine("\nPlease enter the parameters:");
                        }
                        var sb = new System.Text.StringBuilder();
                        while (true)
                        {
                            string input = Console.ReadLine();
                            sb.AppendLine(input);
                            if (Console.ReadKey().Key == ConsoleKey.Enter)
                                break;
                        }
                        //args = sb.ToString().Replace("\r\n", " ").Split(" ").Where(x => !string.IsNullOrEmpty(x)).ToArray();
                        args = ParseArguments(sb.ToString());// sb.ToString().Replace("\r\n", " ").Split(" ").Where(x => !string.IsNullOrEmpty(x)).ToArray();
                    }

                    string[] allArgs = args;
                    string[] commandLineArgs = allArgs.Length > 1 && !allArgs[0].StartsWith("-")
                        ? allArgs[1..]
                        : allArgs;

                    if (commandLineArgs.Length == 0)
                    {
                        Console.WriteLine($"Select language: 1. English; 2. Chinese;");
                        int selectLang = int.TryParse(Console.ReadLine(), out int l) ? l : 0;
                        _lang = selectLang == 2 ? "zh" : "en";

                        Console.WriteLine($"Select input type: 1. File (call AutoPunctuationWithFile); 2. Text (call AutoPunctuationWithText);");
                        int inputType = int.TryParse(Console.ReadLine(), out int t) ? t : 0;

                        commandLineArgs = inputType == 2
                            ? new[] { "-text", "请输入文本内容..." }
                            : new string[0];
                    }

                    var envConfig = new Dictionary<string, string?>
                    {
                        { "modelBasePath", Environment.GetEnvironmentVariable(EnvModelBasePath) ?? "" },
                        { "modelName", Environment.GetEnvironmentVariable(EnvModelName) ?? "default-model" },
                        { "modelAccuracy", Environment.GetEnvironmentVariable(EnvModelAccuracy) ?? "int8" },
                        { "threads", Environment.GetEnvironmentVariable(EnvThreads) ?? "2" },
                        { "text", null }, // 文本输入仅通过命令行，不使用环境变量
                        { "files", null },
                        { "splitSize", "15" } // 默认分段大小
                    };

                    var appConfig = ParseArgs(commandLineArgs, envConfig);
                    ExecuteAutoPunctuation(appConfig);
                }
                catch (ArgumentException ex)
                {
                    Console.WriteLine($"Parameter error: {ex.Message}");
                    PrintUsage();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Program error: {ex.Message}");
                }
                args = Array.Empty<string>();
            }
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
        /// 解析命令行参数，区分文件和文本输入，新增splitSize参数处理
        /// </summary>
        private static Dictionary<string, object> ParseArgs(string[] args, Dictionary<string, string?> envConfig)
        {
            var config = new Dictionary<string, object>
            {
                { "modelBasePath", envConfig["modelBasePath"] },
                { "modelName", envConfig["modelName"] },
                { "modelAccuracy", envConfig["modelAccuracy"] },
                { "threads", int.Parse(envConfig["threads"]!) },
                { "files", Array.Empty<string>() },
                { "text", envConfig["text"] },
                { "splitSize", int.Parse(envConfig["splitSize"]!) }, // 新增分段大小参数
                { "inputType", "none" } // 标识输入类型：files/text
            };

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i].ToLower())
                {
                    case "-base":
                        if (i + 1 < args.Length)
                            config["modelBasePath"] = args[++i];
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
                            throw new ArgumentException("Threads must be a valid integer");
                        break;
                    case "-split": // 新增split参数处理
                        if (i + 1 < args.Length && int.TryParse(args[++i], out int splitSize))
                            config["splitSize"] = splitSize;
                        else
                            throw new ArgumentException("Split size must be a valid integer");
                        break;
                    case "-files":
                        var files = new List<string>();
                        while (++i < args.Length && !args[i].StartsWith("-"))
                            files.Add(args[i].Trim('\"'));
                        config["files"] = files.ToArray();
                        config["inputType"] = "files";
                        i--;
                        break;
                    case "-text":
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("-"))
                        {
                            config["text"] = args[++i];
                            config["inputType"] = "text";
                        }
                        else
                            throw new ArgumentException("Missing content for -text parameter");
                        break;
                    default:
                        throw new ArgumentException($"Unknown parameter: {args[i]}");
                }
            }

            // 验证输入类型
            if (config["inputType"].ToString() == "none")
                throw new ArgumentException("Must specify either -files (for AutoPunctuationWithFile) or -text (for AutoPunctuationWithText)");

            return config;
        }

        /// <summary>
        /// 执行入口：根据输入类型调用对应方法，使用新增的splitSize参数
        /// </summary>
        private static void ExecuteAutoPunctuation(Dictionary<string, object> config)
        {
            // 解析通用参数
            string modelBasePath = config["modelBasePath"].ToString()!.ToLower();
            string modelName = config["modelName"].ToString()!;
            string modelAccuracy = config["modelAccuracy"].ToString()!;
            int threads = (int)config["threads"];
            int splitSize = (int)config["splitSize"]; // 从配置获取分段大小
            string inputType = config["inputType"].ToString()!;

            // 处理默认模型名称
            if (modelName == "default-model")
                modelName = _defaultModelName["alicttransformer"];

            // 打印配置信息
            PrintConfig(modelBasePath, modelName, modelAccuracy, threads, splitSize, inputType, config);

            try
            {
                // 根据输入类型调用对应方法
                if (inputType == "files")
                {
                    string[] files = (string[])config["files"];
                    AutoPunctuationWithFile(
                        modelName: modelName,
                        modelAccuracy: modelAccuracy,
                        threadsNum: threads,
                        filePaths: files,
                        modelBasePath: modelBasePath,
                        splitSize: splitSize
                    );
                }
                else if (inputType == "text")
                {
                    string? text = config["text"] as string;
                    AutoPunctuationWithText(
                        modelName: modelName,
                        modelAccuracy: modelAccuracy,
                        threadsNum: threads,
                        str: text,
                        modelBasePath: modelBasePath,
                        splitSize: splitSize
                    );
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Processing failed: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
        }

        // 辅助方法：获取完整模型路径
        private static string GetFullModelPath(string? modelBasePath, string modelName)
        {
            return string.IsNullOrEmpty(modelBasePath)
                ? Path.Combine(applicationBase, modelName)
                : Path.Combine(modelBasePath, modelName);
        }

        // 辅助方法：模拟添加标点（实际场景替换为模型调用）
        private static string AddPunctuation(string content, int splitSize)
        {
            if (string.IsNullOrEmpty(content))
                return "";
            // 按splitSize分段，每段末尾添加标点
            var segments = new List<string>();
            for (int i = 0; i < content.Length; i += splitSize)
            {
                int len = Math.Min(splitSize, content.Length - i);
                segments.Add(content.Substring(i, len));
            }
            return string.Join("。", segments) + "。"; // 示例：中文标点
        }

        // 打印配置信息（移除batch相关，新增splitSize显示）
        private static void PrintConfig(
            string modelBasePath, string modelName,
            string modelAccuracy, int threads, int splitSize,
            string inputType, Dictionary<string, object> config)
        {
            if (_lang == "zh")
            {
                Console.WriteLine("===== 自动标点配置 =====");
                Console.WriteLine($"模型目录: {modelBasePath}");
                Console.WriteLine($"模型名称: {modelName}");
                Console.WriteLine($"模型精度: {modelAccuracy}");
                Console.WriteLine($"线程数: {threads}");
                Console.WriteLine($"分段大小: {splitSize}"); // 新增显示
                Console.WriteLine($"处理方法: {(inputType == "files" ? "AutoPunctuationWithFile" : "AutoPunctuationWithText")}");
                Console.WriteLine($"输入内容: {(inputType == "files" ? string.Join(", ", (string[])config["files"]) : config["text"])}");
                Console.WriteLine("======================");
            }
            else
            {
                Console.WriteLine("===== Auto Punctuation Configuration =====");
                Console.WriteLine($"Model Directory: {modelBasePath}");
                Console.WriteLine($"Model Name: {modelName}");
                Console.WriteLine($"Precision: {modelAccuracy}");
                Console.WriteLine($"Threads: {threads}");
                Console.WriteLine($"Split Size: {splitSize}"); // 新增显示
                Console.WriteLine($"Method: {(inputType == "files" ? "AutoPunctuationWithFile" : "AutoPunctuationWithText")}");
                Console.WriteLine($"Input: {(inputType == "files" ? string.Join(", ", (string[])config["files"]) : config["text"])}");
                Console.WriteLine("==================================");
            }
        }

        // 打印使用说明（更新参数说明）
        private static void PrintUsage()
        {
            if (_lang == "zh")
            {
                Console.WriteLine("\n使用说明: 调用 AutoPunctuationWithFile 或 AutoPunctuationWithText 方法");
                Console.WriteLine("必选输入参数（二选一）:");
                Console.WriteLine("    -files <文件1> <文件2>    调用 AutoPunctuationWithFile（处理文件）");
                Console.WriteLine("    -text <文本内容>         调用 AutoPunctuationWithText（处理文本）");
                Console.WriteLine("可选参数:");
                Console.WriteLine($"  -base <模型目录>          模型存放目录（环境变量: {EnvModelBasePath}）");
                Console.WriteLine($"  -model <名称>            模型名称（默认: default-model，环境变量: {EnvModelName}）");
                Console.WriteLine($"  -accuracy <fp32/int8>    模型精度（默认: int8，环境变量: {EnvModelAccuracy}）");
                Console.WriteLine($"  -threads <数量>          线程数（默认: 2，环境变量: {EnvThreads}）");
                Console.WriteLine($"  -split <大小>            分段大小（默认: 15）"); // 新增说明
                Console.WriteLine("\n示例1（调用文件处理方法）:");
                Console.WriteLine("  AliCTTransformerPunc.Examples.exe -files ./test1.txt ./test2.txt -model my-model -split 20");
                Console.WriteLine("\n示例2（调用文本处理方法）:");
                Console.WriteLine("  AliCTTransformerPunc.Examples.exe -text \"这是一段测试文本\" -accuracy fp32 -threads 4");
            }
            else
            {
                Console.WriteLine("\nUsage: Call AutoPunctuationWithFile or AutoPunctuationWithText");
                Console.WriteLine("Required input parameters (choose one):");
                Console.WriteLine("    -files <file1> <file2>    Call AutoPunctuationWithFile (process files)");
                Console.WriteLine("    -text <content>          Call AutoPunctuationWithText (process text)");
                Console.WriteLine("Optional parameters:");
                Console.WriteLine($"  -base <model dir>         Model directory (env: {EnvModelBasePath})");
                Console.WriteLine($"  -model <name>             Model name (default: default-model, env: {EnvModelName})");
                Console.WriteLine($"  -accuracy <fp32/int8>     Precision (default: int8, env: {EnvModelAccuracy})");
                Console.WriteLine($"  -threads <count>          Thread count (default: 2, env: {EnvThreads})");
                Console.WriteLine($"  -split <size>             Split size (default: 15)"); // 新增说明
                Console.WriteLine("\nExample 1 (call file method):");
                Console.WriteLine("  AliCTTransformerPunc.Examples.exe -files ./test1.txt ./test2.txt -model my-model -split 20");
                Console.WriteLine("\nExample 2 (call text method):");
                Console.WriteLine("  AliCTTransformerPunc.Examples.exe -text \"This is a test\" -accuracy fp32 -threads 4");
            }
        }
    }
}