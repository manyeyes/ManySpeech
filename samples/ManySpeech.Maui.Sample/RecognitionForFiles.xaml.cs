using ManySpeech.Maui.Sample.Utils;
using Microsoft.Maui.Storage;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class RecognitionForFiles : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.ApplicationBase, "AllModels");
    private string[] _downloadHostParams = new string[] { "https://www.modelscope.cn/models", "master" };
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 sensevoice, paraformer onnx 离线模型（非流式模型）
    // 3.设置 _modelName 值，_modelName = [模型名称]
    private string _modelName = "sensevoice-small-int8-onnx";
    // 需要检查的文件 <文件名, hash>
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        {"model.int8.onnx",""},
        {"am.mvn","" },
        {"asr.json","" },
        {"tokens.txt","" }
    };
    private AliParaformerAsr.OfflineRecognizer _offlineRecognizer = null;

    public RecognitionForFiles()
    {
        InitializeComponent();
        //GetDownloadState();
    }

    private async void OnDownLoadCheckClicked(object sender, EventArgs e)
    {
        BtnDownLoadCheck.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DownloadCheck();
        });
        BtnDownLoadCheck.IsEnabled = true;
    }

    private async void OnDownLoadModelsClicked(object sender, EventArgs e)
    {
        BtnDownLoadModels.IsEnabled = false;
        DownloadProgressBar.Progress = 0 / 100.0;
        DownloadProgressLabel.Text = "";
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DownloadModels();
        });
        BtnDownLoadModels.IsEnabled = true;
    }

    private async void OnDeleteModelsClicked(object sender, EventArgs e)
    {
        BtnDeleteModels.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DeleteModels();
        });
        BtnDeleteModels.IsEnabled = true;
    }

    private async void DownloadModels()
    {
        DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                 }));
        GitHelper gitHelper = new GitHelper(this.DownloadDisplay);
        await Task.Run(() => gitHelper.ProcessCloneModel(_modelBase, _modelName));
    }

    private async void DeleteModels()
    {
        GitHelper gitHelper = new GitHelper(this.DownloadDisplay);
        await Task.Run(() => gitHelper.DeleteModels(_modelBase, _modelName));
    }

    async void GetDownloadState()
    {
        DownloadHelper downloadHelper = new DownloadHelper();
        bool state = downloadHelper.GetDownloadState(_modelFiles, _modelBase, _modelName);
        ModelStatusLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 if (state)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                 }

                             }));
    }

    async void DownloadCheck()
    {
        DownloadHelper downloadHelper = new DownloadHelper(_modelBase, this.DownloadDisplay);
        bool state = downloadHelper.GetDownloadState(_modelFiles, _modelBase, _modelName);
        ModelStatusLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 ModelStatusLabel.IsVisible = true;
                                 if (state)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                 }

                             }));
        if (!state)
        {
            DownloadResultsLabel.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 DownloadResultsLabel.IsVisible = true;
                                 DownloadResultsLabel.Text = "";
                             }));
            foreach (var modelFile in _modelFiles)
            {
                downloadHelper.DownloadCheck(modelFile.Key, _modelBase, _modelName, modelFile.Value);
            }
        }
        else
        {
            DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                 }));
        }
    }

    private void DownloadDisplay(int progress, DownloadState downloadState, string filename, string msg = "")
    {
        if (progress == 0 && downloadState == DownloadState.inprogres)
        {
            DownloadProgressLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.IsVisible = true;
                                     DownloadProgressLabel.Text = msg;
                                 }));
        }
        else
        {
            switch (downloadState)
            {
                case DownloadState.inprogres:
                    DownloadProgressBar.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressBar.Progress = progress / 100.0;
                                 }));
                    DownloadProgressLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.Text = $"File: {filename}, downloading, progress: {progress}%\n";
                                 }));

                    break;
                case DownloadState.cancelled:
                    DownloadProgressBar.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressBar.Progress = progress / 100.0;
                                 }));
                    DownloadProgressLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.Text = $"File: {filename}, download cancelled\n";
                                 }));
                    break;
                case DownloadState.error:
                    DownloadProgressBar.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressBar.Progress = progress / 100.0;
                                 }));
                    DownloadProgressLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.Text = $"File: {filename}, download failed: {msg}\n";
                                 }));
                    DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadResultsLabel.Text += $"File: {filename}, download failed: {msg}\n";
                                 }));
                    break;
                case DownloadState.completed:
                    DownloadProgressBar.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressBar.Progress = progress / 100.0;
                                 }));
                    DownloadProgressLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.Text = $"File: {filename}, download completed\n";
                                 }));
                    DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadResultsLabel.Text += $"File: {filename}, download completed\n";
                                 }));
                    break;
                case DownloadState.existed:
                    DownloadProgressBar.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressBar.Progress = progress / 100.0;
                                 }));
                    DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadResultsLabel.Text += $"File: {filename}, already exists\n";
                                 }));
                    break;
                case DownloadState.noexisted:
                    DownloadResultsLabel.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.Text += $"File: {filename}, does not exist\n";
                                 }));
                    break;
            }
        }
    }

    //private void DownloadDisplay(int progress, DownloadState downloadState, string filename, string msg = "")
    //{
    //    if (progress == 0 && downloadState == DownloadState.inprogres)
    //    {
    //        DownloadProgressLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.IsVisible = true;
    //                                 DownloadProgressLabel.Text = msg;
    //                             }));
    //    }
    //    else
    //    {
    //        switch (downloadState)
    //        {
    //            case DownloadState.inprogres:
    //                DownloadProgressBar.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressBar.Progress = progress / 100.0;
    //                             }));
    //                DownloadProgressLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.Text = $"File：{filename}，正在下载，进度：{progress}%\n";
    //                             }));

    //                break;
    //            case DownloadState.cancelled:
    //                DownloadProgressBar.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressBar.Progress = progress / 100.0;
    //                             }));
    //                DownloadProgressLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.Text = $"文件：{filename}，下载已取消\n";
    //                             }));
    //                break;
    //            case DownloadState.error:
    //                DownloadProgressBar.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressBar.Progress = progress / 100.0;
    //                             }));
    //                DownloadProgressLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.Text = $"文件：{filename}，下载失败：{msg}\n";
    //                             }));
    //                DownloadResultsLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadResultsLabel.Text += $"文件：{filename}，下载失败：{msg}\n";
    //                             }));
    //                break;
    //            case DownloadState.completed:
    //                DownloadProgressBar.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressBar.Progress = progress / 100.0;
    //                             }));
    //                DownloadProgressLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.Text = $"文件：{filename}，下载完成\n";
    //                             }));
    //                DownloadResultsLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadResultsLabel.Text += $"文件：{filename}，下载完成\n";
    //                             }));
    //                break;
    //            case DownloadState.existed:
    //                DownloadProgressBar.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressBar.Progress = progress / 100.0;
    //                             }));
    //                DownloadResultsLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadResultsLabel.Text += $"文件：{filename}，已存在\n";
    //                             }));
    //                break;
    //            case DownloadState.noexisted:
    //                DownloadResultsLabel.Dispatcher.Dispatch(
    //                         new Action(
    //                             delegate
    //                             {
    //                                 DownloadProgressLabel.IsVisible = false;
    //                                 DownloadResultsLabel.Text += $"文件：{filename}，不存在\n";
    //                             }));
    //                break;
    //        }
    //    }
    //}

    private async void OnBtnRecognitionExampleClicked(object sender, EventArgs e)
    {
        BtnRecognitionExample.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            RecognizerTestFilesByOffline();
        });
        BtnRecognitionExample.IsEnabled = true;
    }

    private async void OnBtnRecognitionFilesClicked(object sender, EventArgs e)
    {
        var customFileType = new FilePickerFileType(
                new Dictionary<DevicePlatform, IEnumerable<string>>
                {
                    { DevicePlatform.iOS, new[] { "public.my.comic.extension" } }, // UTType values
                    { DevicePlatform.Android, new[] { "audio/x-wav" } }, // MIME type
                    { DevicePlatform.WinUI, new[] { ".wav", ".mp3" } }, // file extension
                    { DevicePlatform.Tizen, new[] { "*/*" } },
                    { DevicePlatform.macOS, new[] { "cbr", "cbz" } }, // UTType values
                });

        PickOptions options = new()
        {
            PickerTitle = "Please select a comic file",
            FileTypes = customFileType,
        };


        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            var fileResult = await PickAndShow(options);
            if (fileResult != null)
            {
                string fullpath = fileResult.FullPath;
                List<string> fullpaths = new List<string>();
                fullpaths.Add(fullpath);
                RecognizerFilesByOffline(fullpaths);
            }
        });
    }

    public async Task<FileResult> PickAndShow(PickOptions options)
    {
        try
        {
            var result = await FilePicker.Default.PickAsync(options);
            return result;
        }
        catch (Exception ex)
        {
            // The user canceled or something went wrong
        }

        return null;
    }

    public void CreateDownloadFile(string fileName)
    {

        var downloadFolder = FileSystem.AppDataDirectory + "/Download/";
        Directory.CreateDirectory(downloadFolder);
        var filePath = downloadFolder + fileName;
        File.Create(filePath);
    }

    public AliParaformerAsr.OfflineRecognizer initAliParaformerAsrOfflineRecognizer(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
    {
        if (_offlineRecognizer == null)
        {
            if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
            {
                return null;
            }
            string modelFilePath = modelBasePath + "./" + modelName + "/model.int8.onnx";
            string configFilePath = modelBasePath + "./" + modelName + "/asr.yaml";
            string mvnFilePath = modelBasePath + "./" + modelName + "/am.mvn";
            string tokensFilePath = modelBasePath + "./" + modelName + "/tokens.txt";
            string modelebFilePath = modelBasePath + "./" + modelName + "/model_eb.int8.onnx";
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
                _offlineRecognizer = new AliParaformerAsr.OfflineRecognizer(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, tokensFilePath: tokensFilePath, modelebFilePath: modelebFilePath, hotwordFilePath: hotwordFilePath, threadsNum: threadsNum);
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

    public void RecognizerFilesByOffline(List<string> fullpaths)
    {
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = initAliParaformerAsrOfflineRecognizer(_modelName, _modelBase);
        if (offlineRecognizer == null) { return; }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "Speech recognition in progress, please wait……";
                             }
                             ));
        List<float[]>? samples = new List<float[]>();
        List<string> paths = new List<string>();
        TimeSpan totalDuration = new TimeSpan(0L);
        foreach (string fullpath in fullpaths)
        {
            string mediaFilePath = string.Format(fullpath);
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
                    totalDuration += duration;
                }
            }
        }
        if (samples.Count == 0)
        {
            AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "No media file is read!";
                             }
                             ));
            return;
        }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                                 List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
                                 foreach (var sample in samples)
                                 {
                                     AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                                     stream.AddSamples(sample);
                                     streams.Add(stream);
                                 }
                                 List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                                 int n = 0;
                                 foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
                                 {
                                     string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine("{");
                                     //r.AppendLine($"\"path\": \"{paths[n]}\",");
                                     r.AppendLine($"\"text\": \"{text}\",");
                                     r.AppendLine($"\"tokens\":[{string.Join(",", result?.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                                     r.AppendLine($"\"timestamps\":[{string.Join(",", result?.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                                     r.AppendLine("}\n");
                                     AsrResults.Text = r.ToString();
                                     n++;
                                 }
                                 TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                                 double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                                 double rtf = elapsedMilliseconds / totalDuration.TotalMilliseconds;
                                 AsrResults.Text += string.Format("Recognition elapsed milliseconds:{0}\n", elapsedMilliseconds.ToString());
                                 AsrResults.Text += string.Format("Total duration milliseconds:{0}\n", totalDuration.TotalMilliseconds.ToString());
                                 AsrResults.Text += string.Format("Rtf:{1}\n", "0".ToString(), rtf.ToString());
                             }));

    }

    public void RecognizerTestFilesByOffline(List<float[]>? samples = null)
    {
        AliParaformerAsr.OfflineRecognizer offlineRecognizer = initAliParaformerAsrOfflineRecognizer(_modelName, _modelBase);
        if (offlineRecognizer == null) { return; }
        AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "Speech recognition in progress, please wait ...";
                             }
                             ));
        TimeSpan totalDuration = new TimeSpan(0L);
        List<string> paths = new List<string>();
        try
        {
            if (samples == null)
            {
                samples = new List<float[]>();
                string[]? mediaFilePaths = null;
                if (mediaFilePaths == null || mediaFilePaths.Count() == 0)
                {
                    string fullPath = Path.Combine(_modelBase, _modelName);
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
                            totalDuration += duration;
                        }
                    }
                }
            }
            if (samples.Count == 0)
            {
                AsrResults.Dispatcher.Dispatch(
                         new Action(
                             delegate
                             {
                                 AsrResults.Text = "No media file is read!";
                             }
                             ));
                return;
            }
            AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                                     List<AliParaformerAsr.OfflineStream> streams = new List<AliParaformerAsr.OfflineStream>();
                                     foreach (var sample in samples)
                                     {
                                         AliParaformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                                         stream.AddSamples(sample);
                                         streams.Add(stream);
                                     }
                                     List<AliParaformerAsr.Model.OfflineRecognizerResultEntity> results = offlineRecognizer.GetResults(streams);
                                     int n = 0;
                                     AsrResults.Text = "";
                                     foreach (AliParaformerAsr.Model.OfflineRecognizerResultEntity result in results)
                                     {
                                         AsrResults.Text += n + "." + Path.GetFileName(paths[n]) + "\n";
                                         StringBuilder r = new StringBuilder();
                                         string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
                                         r.AppendLine("{");
                                         //r.AppendLine($"\"path\": \"{paths[n]}\",");
                                         r.AppendLine($"\"text\": \"{text}\",");
                                         r.AppendLine($"\"tokens\":[{string.Join(",", result?.Tokens.Select(x => $"\"{x}\"").ToArray())}],");
                                         r.AppendLine($"\"timestamps\":[{string.Join(",", result?.Timestamps.Select(x => $"[{x.First()},{x.Last()}]").ToArray())}]");
                                         r.AppendLine("}\n");
                                         AsrResults.Text += r.ToString();
                                         n++;
                                     }
                                     TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                                     double elapsedMilliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                                     double rtf = elapsedMilliseconds / totalDuration.TotalMilliseconds;
                                     AsrResults.Text += string.Format("Recognition elapsed milliseconds:{0}\n", elapsedMilliseconds.ToString());
                                     AsrResults.Text += string.Format("Total duration milliseconds:{0}\n", totalDuration.TotalMilliseconds.ToString());
                                     AsrResults.Text += string.Format("Rtf:{1}\n", "0".ToString(), rtf.ToString());
                                 }
                                 ));
        }
        catch (Exception ex)
        {
            AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     AsrResults.Text = ex.Message;
                                 }
                                 ));
        }

    }
}

