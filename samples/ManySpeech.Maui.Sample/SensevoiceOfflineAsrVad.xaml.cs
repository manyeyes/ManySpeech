using ManySpeech.Maui.Sample.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class SensevoiceOfflineAsrVad : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 sensevoice, paraformer offline onnx 离线模型（非流式模型）
    // 3.设置 _modelName 值，_modelName = [模型名称]
    private string _modelName = "sensevoice-small-int8-onnx";
    private string _vadModelName = "alifsmnvad-onnx";
    private Dictionary<string, Dictionary<string, string>> _models;
    OfflineAliParaformerAsrRecognizer? _recognizer;
    public SensevoiceOfflineAsrVad()
    {
        InitializeComponent();
        CheckModels();
        // 如需强制先行检查文件，可填_modelFiles <文件名, hash>
        // hash为空时，仅判断文件是否存在
        _models = new Dictionary<string, Dictionary<string, string>>() {
            {
                _modelName,new Dictionary<string, string>{
                    {"model.int8.onnx",""},
                    {"am.mvn","" },
                    {"asr.json","" },
                    {"tokens.txt","" }
                }
            },
            {_vadModelName,new Dictionary<string, string>()}
        };
        LblTitle.Text = string.Join(", ", _models.Keys.ToArray());
    }

    #region check models, download models, delete models
    private async void OnCheckModelsClicked(object sender, EventArgs e)
    {
        BtnCheckModels.IsEnabled = false;
        BtnCheckModels.Text = "Checking...";
        DownloadResultsLabel.Text = "";
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            CheckModels();
        });
        BtnCheckModels.Text = "Check";
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
            foreach (var modelName in _models.Keys.ToArray())
            {
                await DownloadModel(modelName);
            }
        });
        BtnDownLoadModels.IsEnabled = true;
    }

    private async void OnDeleteModelsClicked(object sender, EventArgs e)
    {
        BtnDeleteModels.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(() =>
        {
            DeleteModels(_models.Keys.ToArray());
        });
        BtnDeleteModels.IsEnabled = true;
    }
    private void CheckModels()
    {
        DownloadHelper downloadHelper = new DownloadHelper(_modelBase, this.DownloadDisplay);
        Dispatcher.Dispatch(
                         new Action(
                             async delegate
                             {
                                 bool status = true;
                                 ModelStatusLabel.IsVisible = true;
                                 foreach (var modelName in _models.Keys.ToArray())
                                 {
                                     var modelFiles = _models[modelName];
                                     bool state = downloadHelper.GetDownloadState(modelFiles, _modelBase, modelName);
                                     if (state)
                                     {
                                         status = status && state;
                                     }
                                     else
                                     {
                                         status = status && state;
                                         bool isDownload = await DisplayAlert("Question?", $"Missing model: {modelName}, will it be automatically downloaded?", "Yes", "No");
                                         if (isDownload)
                                         {
                                             await DownloadModel(modelName);
                                         }
                                     }
                                 }
                                 if (status)
                                 {
                                     ModelStatusLabel.Text = "model is ready";
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                 }
                                 else
                                 {
                                     ModelStatusLabel.Text = "model not ready";
                                     DownloadResultsLabel.IsVisible = true;
                                     DownloadResultsLabel.Text = "";
                                 }
                             }));

    }
    private async Task DownloadModel(string modelName)
    {
        Dispatcher.Dispatch(
                             new Action(
                                 delegate
                                 {
                                     DownloadProgressLabel.IsVisible = false;
                                     DownloadResultsLabel.IsVisible = false;
                                     DownloadResultsLabel.Text = "";
                                 }));
        GitHelper gitHelper = new GitHelper(this.DownloadDisplay);
        await Task.Run(() => gitHelper.ProcessCloneModel(_modelBase, modelName));
    }

    private async void DeleteModels(string[] modelNames)
    {
        GitHelper gitHelper = new GitHelper(this.DownloadDisplay);
        var tasks = new List<Task>();
        foreach (var currentModel in modelNames)
        {
            tasks.Add(Task.Run(() => gitHelper.DeleteModels(_modelBase, currentModel)));
        }
        try
        {
            await Task.WhenAll(tasks);
        }
        catch (AggregateException ex)
        {
            foreach (var innerEx in ex.InnerExceptions)
            {
                ShowTips($"Failed to delete model：{innerEx.Message}");
            }
        }
    }
    #endregion

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
            string methodType = "chunk";
            string outputFormat = "srt"; // txt/srt
            string recognizerType = "offline";
            int threads = 2;
            if (_recognizer == null)
            {
                _recognizer = new OfflineAliParaformerAsrRecognizer();
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
                var timestampsList = new List<List<int[]>>();
                var vadDetector = new AliFsmnVadDetector();
                var vadResult = vadDetector.OfflineDetector(samples.Value.sampleList, _modelBase, _vadModelName, modelAccuracy, threads); // 使用多流模式
                samplesList = vadResult.Select(x => x.Waveform).ToList();
                timestampsList = vadResult.Select(x => x.Segment.Select(y => new int[] { y[0], y[1] }).ToList()).ToList();
                SetOfflineRecognizerCallbackForResult(_recognizer, recognizerType, outputFormat, timestampsList: timestampsList);
                SetOfflineRecognizerCallbackForCompleted(_recognizer);
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
    private void SetOfflineRecognizerCallbackForResult(OfflineAliParaformerAsrRecognizer recognizer, string? recognizerType, string outputFormat = "txt", int startIndex = 0, List<List<int[]>> timestampsList = null)
    {
        List<int> orderIndexList = timestampsList != null ? new int[timestampsList.Count].ToList() : null;
        var timestamps = timestampsList != null ? ConvertHelper.Convert(timestampsList, orderIndexList).ToList() : null;
        int i = startIndex;
        recognizer.ResetRecognitionResultHandlers();
        recognizer.OnRecognitionResult += async result =>
        {
            string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
            if (!string.IsNullOrEmpty(text))
            {
                int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                StringBuilder r = new StringBuilder();
                switch (outputFormat)
                {
                    case "txt":
                        r.Clear();
                        r.AppendLine($"[{recognizerType} Stream {resultIndex}]");
                        r.AppendLine(text);
                        ShowResults($"{r.ToString()}", true);
                        break;
                    case "srt":
                        r.Clear();
                        if (timestamps != null && timestamps.Count > i - startIndex)
                        {
                            var outerIndex = timestamps[i - startIndex].outerIndex;
                            var innerIndex = timestamps[i - startIndex].innerIndex;
                            var timestamp = timestamps[i - startIndex].timestamp;
                            r.AppendLine(resultIndex.ToString());
                            r.AppendLine($"{TimeSpan.FromMilliseconds(timestamp[0]).ToString(@"hh\:mm\:ss\.fff").Replace('.', ',')} -> {TimeSpan.FromMilliseconds(timestamp[1]).ToString(@"hh\:mm\:ss\.fff").Replace('.', ',')}");
                        }
                        else
                        {
                            r.AppendLine(resultIndex.ToString());
                            if (result.Timestamps.Count() > 0)
                            {
                                r.AppendLine($"{TimeSpan.FromMilliseconds(result.Timestamps.First()[0]).ToString(@"hh\:mm\:ss\.fff").Replace('.', ',')}->{TimeSpan.FromMilliseconds(result.Timestamps.Last()[1]).ToString(@"hh\:mm\:ss\.fff").Replace('.', ',')}");
                            }
                        }
                        r.AppendLine(text);
                        ShowResults($"{r.ToString()}", true);
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
            StringBuilder r = new StringBuilder();
            r.AppendLine(string.Format("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString()));
            r.AppendLine(string.Format("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString()));
            r.AppendLine(string.Format("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString()));
            ShowResults($"{r.ToString()}", true);
        };
    }
    #endregion
}
