using ManySpeech.Maui.Sample.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System;
using System.Text;
using static Microsoft.Maui.ApplicationModel.Permissions;

namespace ManySpeech.Maui.Sample;

public partial class FireRedOfflineAsr : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 FireRed 离线模型（非流式模型）
    // 3.设置 _modelName = [模型名称]
    private string _modelName = "fireredasr-aed-large-zh-en-onnx-offline-20250124";
    // 需要检查的文件 <文件名, hash>
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        {"encoder.int8.onnx",""},
        {"decoder.int8.onnx",""},
        {"tokens.txt","" }
    };

    public FireRedOfflineAsr()
    {
        InitializeComponent();
        CheckModels();
    }

    #region event
    private async void OnCheckModelsClicked(object sender, EventArgs e)
    {
        BtnCheckModels.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            CheckModels();
        });
        BtnCheckModels.IsEnabled = true;
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
    #endregion

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

    private void CheckModels()
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
            RecognitionExampleByOffline();
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
            ShowResults("The user canceled or something went wrong:"+ex.ToString());
        }

        return null;
    }

    public void RecognizerFilesByOffline(List<string> fullpaths)
    {
        var recognizer = new OfflineFireRedAsrRecognizer();
        SetOfflineRecognizerCallbackForResult(recognizer, "offline", "text");
        SetOfflineRecognizerCallbackForCompleted(recognizer);
        ShowResults("Speech recognition in progress, please wait ...");
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
            ShowResults("No media file is read!");
            return;
        }
        this.Dispatcher.Dispatch(
                         new Action(
                              async delegate
                             {
                                 ShowResults("");
                                 string modelAccuracy = "int8";
                                 string methodType = "one";
                                 int threads = 2;
                                 var samplesList = new List<List<float[]>>();
                                 samplesList = samples.Select(x => new List<float[]>() { x }).ToList();
                                 try
                                 {
                                     await recognizer.RecognizeAsync(
                                            samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
                                 }
                                 catch (Exception ex)
                                 {
                                     ShowTips(ex.Message);
                                 }
                             }));

    }

    public void RecognitionExampleByOffline(List<float[]>? samples = null)
    {
        var recognizer = new OfflineFireRedAsrRecognizer();
        SetOfflineRecognizerCallbackForResult(recognizer, "offline", "text");
        SetOfflineRecognizerCallbackForCompleted(recognizer);
        if (recognizer == null) { return; }
        ShowResults("Speech recognition in progress, please wait ...");
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
                ShowResults("No media file is read!");
                return;
            }
            this.Dispatcher.Dispatch(
                             new Action(
                                 async delegate
                                 {
                                     ShowResults("");
                                     string modelAccuracy = "int8";
                                     string methodType = "one";
                                     int threads = 2;
                                     var samplesList = new List<List<float[]>>();
                                     samplesList = samples.Select(x => new List<float[]>() { x }).ToList();
                                     try
                                     {
                                         await recognizer.RecognizeAsync(
                                                samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
                                     }
                                     catch (Exception ex)
                                     {
                                         ShowTips(ex.Message);
                                     }
                                 }
                                 ));
        }
        catch (Exception ex)
        {
            ShowResults(ex.Message);
        }

    }

    private void ShowTips(string str)
    {
        this.Dispatcher.Dispatch(
                    new Action(
                        async delegate
                        {
                            await DisplayAlert("Tips", str, "close");
                        }));
    }

    private void ShowResults(string str, bool isAppend = false)
    {
        AsrResults.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     if (isAppend)
                                     {
                                         AsrResults.Text += str;
                                     }
                                     else
                                     {
                                         AsrResults.Text = str;
                                     }

                                 }
                                 ));


    }

    #region callback
    private void SetOfflineRecognizerCallbackForResult(OfflineFireRedAsrRecognizer recognizer, string? recognizerType, string outputFormat = "text")
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
                        this.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine($"[{recognizerType} Stream {resultIndex}]");
                                     r.AppendLine(text);
                                     ShowResults($"{r.ToString()}" + "\r", true);
                                 }
                                 ));
                        await LabelAsrResultsScrollView.ScrollToAsync(0D, (double)ScrollToPosition.End, true);
                        break;
                    case "json":
                        this.Dispatcher.Dispatch(
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
                                     ShowResults($"{r.ToString()}" + "\r", true);
                                 }
                                 ));
                        break;
                }
            }
            i++;
        };
    }
    private void SetOfflineRecognizerCallbackForCompleted(OfflineFireRedAsrRecognizer recognizer)
    {
        recognizer.ResetRecognitionCompletedHandlers();
        recognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
        {
            double elapsedMilliseconds = totalTime.TotalMilliseconds;
            this.Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     StringBuilder r = new StringBuilder();
                                     r.AppendLine(string.Format("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString()));
                                     r.AppendLine(string.Format("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString()));
                                     r.AppendLine(string.Format("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString()));
                                     ShowResults($"{r.ToString()}" + "\r", true);
                                 }
                                 ));
        };
    }
    #endregion
}

