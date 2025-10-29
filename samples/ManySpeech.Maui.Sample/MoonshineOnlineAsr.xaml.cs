using AudioInOut.Base;
using ManySpeech.Maui.Sample.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class MoonshineOnlineAsr : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 moonshine onnx 流式模型
    // 3.设置 _modelName 值，_modelName = [模型名称]
    //private string _modelName = "moonshine-base-en-onnx";
    private string _modelName = "moonshine-tiny-en-onnx";
    private string _vadModelName = "alifsmnvad-onnx";
    private Dictionary<string, Dictionary<string, string>> _models;
    private IRecorder _micCapture;
    private CancellationTokenSource _micCaptureCts = new CancellationTokenSource();
    private OnlineMoonshineAsrRecognizer? _recognizer;

    public MoonshineOnlineAsr(IRecorder micCapture)
    {
        InitializeComponent();
        CheckModels();
        _micCapture = micCapture;
        // 如需强制先行检查文件，可填_modelFiles <文件名, hash>
        // hash为空时，仅判断文件是否存在
        _models = new Dictionary<string, Dictionary<string, string>>() {
            {_modelName,new Dictionary<string, string>()},
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

    private async void OnBtnRecognitionMicStartClicked(object sender, EventArgs e)
    {
        ResetComponent();
        BtnRecognitionMicStart.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            // 麦克风识别，参数: -method chunk
            if (_micCapture == null)
            {
                return;
            }
            try
            {
                _micCaptureCts = new CancellationTokenSource();
                await _micCapture.StartCapture();

                string recognizerType = "online";
                string modelAccuracy = "int8";
                int threads = 2;
                if (_recognizer == null)
                {
                    _recognizer = new OnlineMoonshineAsrRecognizer();
                    SetOnlineRecognizerCallbackForResult(_recognizer, recognizerType);
                }
                while (!_micCaptureCts.Token.IsCancellationRequested)
                {
                    var micChunk = await _micCapture.GetNextMicChunkAsync(_micCaptureCts.Token);
                    if (micChunk == null) continue;
                    if (micChunk != null)
                    {
                        await _recognizer.RecognizeAsync(
                        micChunk, _modelBase, _modelName, modelAccuracy, "chunk", threads);
                    }
                }
                ShowTips($"[{DateTime.Now:HH:mm:ss}] Real-time recognition completed");
            }
            catch (OperationCanceledException)
            {
                ShowTips($"[{DateTime.Now:HH:mm:ss}] Real-time recognition canceled by user");
            }
            finally
            {
                if (_recognizer != null)
                {
                    _recognizer.Dispose();
                    _recognizer = null;
                }
            }

        });
        BtnRecognitionMicStart.IsEnabled = false;
        BtnRecognitionMicStart.Background = new SolidColorBrush(Colors.Gray);
        BtnRecognitionMicStart.TextColor = Colors.WhiteSmoke;
    }

    private async void OnBtnRecognitionMicStopClicked(object sender, EventArgs e)
    {
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(() =>
        {
            _micCapture.StopCapture();
            _micCaptureCts.Cancel();
        });
        this.BtnRecognitionMicStart.IsEnabled = true;
        BtnRecognitionMicStart.Background = default;
        BtnRecognitionMicStart.TextColor = default;
    }

    private async void OnBtnRecognitionClearClicked(object sender, EventArgs e)
    {
        ClearLogs();
        ClearResults();
    }

    private async void OnBtnRecognitionExampleClicked(object sender, EventArgs e)
    {
        BtnRecognitionExample.IsEnabled = false;
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            await RecognizerFilesByOnline();
        });
        BtnRecognitionExample.IsEnabled = true;
    }

    private async void OnBtnRecognitionFilesClicked(object sender, EventArgs e)
    {
        ResetComponent();
        var customFileType = new FilePickerFileType(
                new Dictionary<DevicePlatform, IEnumerable<string>>
                {
                    { DevicePlatform.iOS, new[] { "public.my.comic.extension" } }, // UTType values
                    { DevicePlatform.Android, new[] { "audio/*" } }, // MIME type  audio/x-wav
                    { DevicePlatform.WinUI, new[] { ".wav", ".mp3", ".wma", ".ape", ".flac", ".ogg", ".acc",".aac", "aif","aifc","aiff","als","au","awb","es","esl","imy", "audio", ".mp4", "mpg","mpeg"," avi"," rm"," rmvb"," mov"," wmv"," asf", "asx","wvx","mpe","mpa","gdf","3gp","flv","vob","mkv","swf" } }, // file extension
                    { DevicePlatform.Tizen, new[] { "*/*" } },
                    { DevicePlatform.macOS, new[] { ".wav", ".mp3", ".mp4", ".acc" } }, // UTType values
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
                await RecognizerFilesByOnline(fullpaths);
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

    private async Task RecognizerFilesByOnline(List<string>? fullpaths = null)
    {
        try
        {
            // 文件识别 -method one/batch/chunk
            string[] files = !fullpaths?.Any() ?? true ? SampleHelper.GetPaths(_modelBase, _modelName) : fullpaths.ToArray();
            if (files.Length == 0) throw new Exception("No input files found");


            string recognizerType = "online";
            string modelAccuracy = "int8";
            int threads = 2;
            string methodType = "chunk"; // one/batch
            if (_recognizer == null)
            {
                _recognizer = new OnlineMoonshineAsrRecognizer();
                SetOnlineRecognizerCallbackForResult(_recognizer, recognizerType);
            }
            TimeSpan totalDuration = TimeSpan.Zero;
            int tailLength = 18;
            var chunkSamples = SampleHelper.GetChunkSampleFormFile(files, ref totalDuration, chunkSize: 1600 * 6, tailLength: tailLength);
            if (!chunkSamples.HasValue)
            {
                ShowResults("Failed to read audio files");
                return;
            }
            if (methodType == "chunk")
            {
                foreach (var streamSamples in chunkSamples.Value.samplesList)
                {
                    foreach (var sampleChunk in streamSamples)
                    {
                        var chunk = new List<List<float[]>> { new List<float[]> { sampleChunk } };
                        await _recognizer.RecognizeAsync(
                            chunk, _modelBase, _modelName, modelAccuracy, methodType, threads);
                    }
                }
            }
            else
            {
                await _recognizer.RecognizeAsync(
                            chunkSamples.Value.samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
            }
        }
        catch (Exception ex)
        {
            ShowTips(ex.Message);
        }
    }

    private async void ShowResults(string str, bool isAppend = true)
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
        Dispatcher.Dispatch(
                    new Action(
                        async delegate
                        {
                            await DisplayAlert("Tips", str, "close");
                        }));
    }

    private async void OnShowLogsClicked(object sender, EventArgs e)
    {
    }
    private void OnEditAsrResultsClicked(object sender, EventArgs e)
    {
        EditorResults.Text = LblResults.Text;
        EditorResults.IsVisible = true;
        EditorResults.HeightRequest = LblResults.Height;
        LblResults.IsVisible = false;
        BtnEditAsrResults.IsVisible = false;
        BtnEditedAsrResults.IsVisible = true;
    }

    private void OnEditedAsrResultsClicked(object sender, EventArgs e)
    {
        LblResults.Text = EditorResults.Text;
        EditorResults.IsVisible = false;
        LblResults.IsVisible = true;
        BtnEditAsrResults.IsVisible = true;
        BtnEditedAsrResults.IsVisible = false;
    }

    private void ResetComponent()
    {
        ClearResults();
        ClearLogs();
        LblResults.Text = "";
        EditorResults.Text = "";
        EditorResults.IsVisible = false;
        LblResults.IsVisible = true;
        BtnEditAsrResults.IsVisible = true;
        BtnEditedAsrResults.IsVisible = false;
        SetOnlineRecognizerCallbackForResult(_recognizer);
    }
    #region callback  

    private async void SetOnlineRecognizerCallbackForResult(OnlineMoonshineAsrRecognizer recognizer, string? recognizerType = "online")
    {
        if (recognizer == null)
        {
            return;
        }
        SortedDictionary<int, string> _results = new SortedDictionary<int, string>();
        int i = 0;
        recognizer.ResetRecognitionResultHandlers();
        recognizer.OnRecognitionResult += async result =>
        {
            string? text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
            if (!string.IsNullOrEmpty(text))
            {
                int resultIndex = recognizerType == "offline" ? i : result.Index + 1;
                _results[resultIndex] = text;
                StringBuilder r = new StringBuilder();
                foreach (var item in _results)
                {
                    r.AppendLine($"[{recognizerType} Stream {item.Key}]");
                    r.AppendLine(item.Value);
                }
                ShowResults($"{r.ToString()}", false);
            }
            i++;
        };
    }
    private void SetOnlineRecognizerCallbackForCompleted(OnlineMoonshineAsrRecognizer recognizer)
    {
        recognizer.ResetRecognitionCompletedHandlers();
        recognizer.OnRecognitionCompleted += (totalTime, totalDuration, processedCount, sample) =>
        {
            double elapsedMilliseconds = totalTime.TotalMilliseconds;
            StringBuilder r = new StringBuilder();
            r.AppendLine(string.Format("Recognition elapsed milliseconds:{0}", elapsedMilliseconds.ToString()));
            r.AppendLine(string.Format("Total duration milliseconds:{0}", totalDuration.TotalMilliseconds.ToString()));
            r.AppendLine(string.Format("Rtf:{1}", "0".ToString(), (elapsedMilliseconds / totalDuration.TotalMilliseconds).ToString()));
            ShowResults($"{r.ToString()}");
        };
    }

    #endregion
}

