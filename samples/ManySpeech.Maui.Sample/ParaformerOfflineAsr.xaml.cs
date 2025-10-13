using ManySpeech.Maui.Sample.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class ParaformerOfflineAsr : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 sensevoice, paraformer onnx 离线模型（非流式模型）
    // 3.设置 _modelName 值，_modelName = [模型名称]
    private string _modelName = "paraformer-seaco-large-zh-timestamp-int8-onnx-offline";
    // 需要检查的文件 <文件名, hash>
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        {"model.int8.onnx",""},
        {"am.mvn","" },
        {"asr.json","" },
        {"tokens.txt","" }
    };

    public ParaformerOfflineAsr()
    {
        InitializeComponent();
        DownloadCheck();
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
        await taskFactory.StartNew(() =>
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

    private void DownloadCheck()
    {
        DownloadHelper downloadHelper = new DownloadHelper(_modelBase, this.DownloadDisplay);
        ModelStatusLabel.Dispatcher.Dispatch(
                         new Action(
                             async delegate
                             {
                                 ModelStatusLabel.IsVisible = true;
                                 bool state = downloadHelper.GetDownloadState(_modelFiles, _modelBase, _modelName);
                                 if (state)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                     DownloadResultsLabel.IsVisible = true;
                                     DownloadResultsLabel.Text = "";
                                     bool isDownload = await DisplayAlert("Question?", "Missing model, will it be automatically downloaded?", "Yes", "No");
                                     if (isDownload)
                                     {
                                         DownloadModels();
                                     }
                                 }
                             }));

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
        

    public void RecognizerFilesByOffline(List<string> fullpaths)
    {
        var recognizer = new OfflineAliParaformerAsrRecognizer();
        SetOfflineRecognizerCallbackForResult(recognizer, "offline", "text");
        SetOfflineRecognizerCallbackForCompleted(recognizer);
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
                              async delegate
                             {
                                 AsrResults.Text = "";
                                 string modelAccuracy = "int8";
                                 string methodType = "one";
                                 int threads = 2;
                                 var samplesList = new List<List<float[]>>();
                                 samplesList = samples.Select(x => new List<float[]>() { x }).ToList();
                                 await recognizer.RecognizeAsync(
                                        samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
                             }));

    }

    public void RecognizerTestFilesByOffline(List<float[]>? samples = null)
    {
        var recognizer = new OfflineAliParaformerAsrRecognizer();
        SetOfflineRecognizerCallbackForResult(recognizer, "offline", "text");
        SetOfflineRecognizerCallbackForCompleted(recognizer);
        if (recognizer == null) { return; }
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
                                 async delegate
                                 {
                                     AsrResults.Text = "";
                                     string modelAccuracy = "int8";
                                     string methodType = "one";
                                     int threads = 2;
                                     var samplesList = new List<List<float[]>>();
                                     samplesList = samples.Select(x => new List<float[]>() { x }).ToList();
                                     await recognizer.RecognizeAsync(
                                            samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
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

    #region callback
    private void SetOfflineRecognizerCallbackForResult(OfflineAliParaformerAsrRecognizer recognizer, string? recognizerType, string outputFormat = "text")
    {
        int i = 0;
        recognizer.ResetRecognitionResultHandlers();
        recognizer.OnRecognitionResult += async result =>
        {
            string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
            if (!string.IsNullOrEmpty(text))
            {
                int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                switch (outputFormat)
                {
                    case "text":
                        AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine($"[{recognizerType} Stream {resultIndex}]");
                                     r.AppendLine(text);
                                     AsrResults.Text += $"{r.ToString()}" + "\r";
                                 }
                                 ));
                        await LabelAsrResultsScrollView.ScrollToAsync(0D, (double)ScrollToPosition.End, true);
                        break;
                    case "json":
                        AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine($"[{recognizerType} Stream {resultIndex}]");
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
                                     AsrResults.Text += $"{r.ToString()}" + "\r";
                                 }
                                 ));
                        break;
                }
            }
            i++;
        };
    }
    private void SetOfflineRecognizerCallbackForCompleted(OfflineAliParaformerAsrRecognizer recognizer)
    {
        recognizer.ResetRecognitionCompletedHandlers();
        recognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
        {
            double elapsedMilliseconds = totalTime.TotalMilliseconds;
            AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine(string.Format("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString()));
                                     r.AppendLine(string.Format("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString()));
                                     r.AppendLine(string.Format("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString()));
                                     AsrResults.Text += $"{r.ToString()}" + "\r";
                                 }
                                 ));
        };
    }
    #endregion
}

