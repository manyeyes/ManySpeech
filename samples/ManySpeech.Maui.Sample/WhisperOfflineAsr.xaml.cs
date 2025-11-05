using ManySpeech.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class WhisperOfflineAsr : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 whisper onnx 离线模型（非流式模型）
    // 3.设置 _modelName 值，_modelName = [模型名称]
    private string _modelName = "whisper-tiny-onnx";
    // 如需强制先行检查文件，可填_modelFiles <文件名, hash>
    // hash为空时，仅判断文件是否存在
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        {"encoder.int8.onnx",""},
        {"decoder.int8.onnx","" },
        {"conf.json","" }
    }; 
    OfflineWhisperAsrRecognizer? _recognizer;

    public WhisperOfflineAsr()
    {
        InitializeComponent();
        CheckModels();
        LblTitle.Text = _modelName;
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
            await RecognizerFilesByOffline();
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
    private async void OnBtnRecognitionClearClicked(object sender, EventArgs e)
    {
        ClearLogs();
        ClearResults();
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
            ShowResults("The user canceled or something went wrong:" + ex.ToString());
        }

        return null;
    }

    public async Task RecognizerFilesByOffline(List<string>? fullpaths = null)
    {
        try
        {
            string[] files = !fullpaths?.Any() ?? true ? SampleHelper.GetPaths(_modelBase, _modelName) : fullpaths.ToArray();
            if (files.Length == 0)
            {
                ShowResults("No input files found");
                return;
            }
            string modelAccuracy = "int8";
            string methodType = "one";// 文件识别 -method one/batch
            int threads = 2;
            if (_recognizer == null)
            {
                _recognizer = new OfflineWhisperAsrRecognizer();
                SetOfflineRecognizerCallbackForResult(_recognizer, "offline");
                SetOfflineRecognizerCallbackForCompleted(_recognizer);
            }
            if (_recognizer == null) { return; }
            ShowResults("Speech recognition in progress, please wait ...");
            TimeSpan totalDuration = TimeSpan.Zero;
            int tailLength = 6;
            var samples = SampleHelper.GetSampleFormFile(files, ref totalDuration);
            if (!samples.HasValue)
            {
                ShowResults("Failed to read audio files");
                return;
            }
            else
            {
                if (samples.Value.sampleList.Count == 0)
                {
                    ShowResults("No media file is read!");
                    return;
                }
                var samplesList = new List<List<float[]>>();
                samplesList = samples.Value.sampleList.Select(x => new List<float[]>() { x }).ToList();
                await _recognizer.RecognizeAsync(
                           samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
            }
        }
        catch (Exception ex)
        {
            ShowTips(ex.Message);
        }

    }


    private void ShowResults(string str, bool isAppend = true)
    {
        Dispatcher.Dispatch(
                    new Action(
                        async delegate
                        {
                            if (isAppend)
                            {
                                LblResults.Text += str + "\n";
                            }
                            else
                            {
                                LblResults.Text = str + "\n";
                            }
                            await Task.Delay(100);
                            await ScrollViewLabelResults.ScrollToAsync(0, ScrollViewLabelResults.ContentSize.Height, true);
                        }
                        ));
    }
    private void ClearResults()
    {
        Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            LblResults.Text = "";
                        }
                        ));
    }
    private void ShowLogs(string str, bool isAppend = true)
    {
        Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            if (BtnShowLogs.IsEnabled == false)
                            {
                                BtnShowLogs.IsEnabled = true;
                            }
                        }));
    }
    private void ClearLogs()
    {
        Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            if (BtnShowLogs.IsEnabled == true)
                            {
                                BtnShowLogs.IsEnabled = false;
                            }
                        }));
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
    private async void OnShowLogsClicked(object sender, EventArgs e)
    {
    }
    private void OnEditResultsClicked(object sender, EventArgs e)
    {
        EditorResults.Text = LblResults.Text;
        EditorResults.IsVisible = true;
        EditorResults.HeightRequest = LblResults.Height;
        LblResults.IsVisible = false;
        BtnEditResults.IsVisible = false;
        BtnEditedResults.IsVisible = true;
    }

    private void OnEditedResultsClicked(object sender, EventArgs e)
    {
        LblResults.Text = EditorResults.Text;
        EditorResults.IsVisible = false;
        LblResults.IsVisible = true;
        BtnEditResults.IsVisible = true;
        BtnEditedResults.IsVisible = false;
    }

    #region callback
    private void SetOfflineRecognizerCallbackForResult(OfflineWhisperAsrRecognizer recognizer, string? recognizerType)
    {
        int i = 0;
        recognizer.ResetRecognitionResultHandlers();
        recognizer.OnRecognitionResult += async result =>
        {
            string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
            if (!string.IsNullOrEmpty(text))
            {
                int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                StringBuilder r = new StringBuilder();
                r.AppendLine($"[{recognizerType} Stream {resultIndex}]");
                r.AppendLine(text);
                ShowResults($"{r.ToString()}", true);
            }
            i++;
        };
    }
    private void SetOfflineRecognizerCallbackForCompleted(OfflineWhisperAsrRecognizer recognizer)
    {
        recognizer.ResetRecognitionCompletedHandlers();
        recognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
        {
            double elapsedMilliseconds = totalTime.TotalMilliseconds;
            StringBuilder r = new StringBuilder();
            r.AppendLine(string.Format("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString()));
            r.AppendLine(string.Format("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString()));
            r.AppendLine(string.Format("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString()));
            ShowResults($"{r.ToString()}", true);
        };
    }
    #endregion
}

