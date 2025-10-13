using ManySpeech.Maui.Sample.SpeechProcessing;
using ManySpeech.Maui.Sample.Utils;
using PreProcessUtils;
using System.Text;

namespace ManySpeech.Maui.Sample;

public partial class RecognitionForOnline : ContentPage
{
    private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
    // 如何使用其他模型
    // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
    // 2.搜索 sensevoice, paraformer onnx 离线模型（非流式模型）
    // 3.设置 _modelName 值，_modelName = [模型名称]
    private string _modelName = "paraformer-large-zh-en-int8-onnx-online";
    // 需要检查的文件 <文件名, hash>
    private Dictionary<string, string> _modelFiles = new Dictionary<string, string>() {
        {"encoder.int8.onnx",""},
        {"decoder.int8.onnx",""},
        {"am.mvn","" },
        {"asr.json","" },
        {"tokens.txt","" }
    };
    private AudioInOut.Base.IRecorder _micCapture;

    public RecognitionForOnline()
    {
        InitializeComponent();
        DownloadCheck();
    }

    private async void OnDownLoadCheckClicked(object sender, EventArgs e)
    {
        BtnDownLoadCheck.IsEnabled = false;
        BtnDownLoadCheck.Text = "Checking...";
        DownloadResultsLabel.Text = "";
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            DownloadCheck();
        });
        BtnDownLoadCheck.Text = "Check";
        BtnDownLoadCheck.IsEnabled = true;
    }

    //private async void OnDownLoadModelsClicked(object sender, EventArgs e)
    //{
    //    BtnDownLoadModels.IsEnabled = false;
    //    DownloadProgressBar.Progress = 0 / 100.0;
    //    DownloadProgressLabel.Text = "";
    //    TaskFactory taskFactory = new TaskFactory();
    //    await taskFactory.StartNew(async () =>
    //    {
    //        DownloadModels();
    //    });
    //    BtnDownLoadModels.IsEnabled = true;
    //}

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
            int bufferMilliseconds = 200;
            //using var micCapture = new MicAudioCapture(bufferMilliseconds);
            _micCapture = AudioInOut.AudioDeviceFactory.CreateAudioCapture(bufferMilliseconds);
            var cts = new CancellationTokenSource();

            //_ = Task.Run(() =>
            //{
            //    while (!cts.Token.IsCancellationRequested)
            //    {
            //        if (Console.ReadKey(true).Key == ConsoleKey.Escape)
            //        {
            //            cts.Cancel();
            //            _micCapture.StopCapture();
            //            break;
            //        }
            //    }
            //}, cts.Token);
            _micCapture.StartCapture();

            try
            {
                _micCapture.StartCapture();
                string recognizerType = "online";
                string outputFormat = "text";
                string modelAccuracy = "int8";
                int threads = 2;
                var recognizer = new OnlineAliParaformerAsrRecognizer();
                SetOnlineRecognizerCallbackForResult(recognizer, recognizerType, outputFormat);
                //SetOnlineRecognizerCallbackForCompleted(recognizer);
                //if (recognizerType == "2pass")
                //{
                //    var recognizer2 = GetOfflineRecognizer(AsrCategory.AliParaformerAsr);
                //    SetRecognizerCallbackForCompleted2Pass(recognizer, recognizer2, _modelBase, _model2Name, modelAccuracy, "chunk", threads);//, outputFormat, _asrCategory.GetDescription()
                //}
                while (!cts.Token.IsCancellationRequested)
                {
                    var micChunk = await _micCapture.GetNextMicChunkAsync(cts.Token);
                    if (micChunk == null) break;

                    await recognizer.RecognizeAsync(
                        micChunk, _modelBase, _modelName, modelAccuracy, "chunk", threads); // methodType chunk(fix)
                }
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Real-time recognition completed");
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Real-time recognition canceled by user");
            }
            finally
            {
                cts.Cancel();
                _micCapture.Dispose();
            }

        });
        BtnRecognitionMicStart.IsEnabled = false;
        BtnRecognitionMicStart.Background = new SolidColorBrush(Colors.Gray);
        BtnRecognitionMicStart.TextColor = Colors.WhiteSmoke;
    }

    private async void OnBtnRecognitionMicStopClicked(object sender, EventArgs e)
    {
        TaskFactory taskFactory = new TaskFactory();
        await taskFactory.StartNew(async () =>
        {
            _micCapture.StopCapture();
        });
        this.BtnRecognitionMicStart.IsEnabled = true;
        BtnRecognitionMicStart.Background = default;
        BtnRecognitionMicStart.TextColor = default;
        //this.BtnRecognitionMicStart.Background = null;
    }

    private async void OnBtnRecognitionMicCloseClicked(object sender, EventArgs e)
    {
        ClearAsrLogs();
        ClearAsrResults();
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
            //if (result != null)
            //{
            //    if (result.FileName.EndsWith("wav", StringComparison.OrdinalIgnoreCase))
            //    {
            //        using var stream = await result.OpenReadAsync();
            //        var image = ImageSource.FromStream(() => stream);
            //    }
            //}

            return result;
        }
        catch (Exception ex)
        {
            // The user canceled or something went wrong
        }

        return null;
    }

    //public void CreateDownloadFile(string fileName)
    //{

    //    var downloadFolder = FileSystem.AppDataDirectory + "/Download/";
    //    Directory.CreateDirectory(downloadFolder);
    //    var filePath = downloadFolder + fileName;
    //    File.Create(filePath);
    //}

    //public async Task TestRecognizerFilesByOnline()
    //{
    //    await RecognizerFilesByOnline();
    //}

    private async Task RecognizerFilesByOnline(List<string>? fullpaths = null)
    {
        // 文件识别 -method one/batch/chunk
        string[] files = !fullpaths?.Any() ?? true ? SampleHelper.GetPaths(_modelBase, _modelName) : fullpaths.ToArray();
        if (files.Length == 0) throw new Exception("No online input files found");

        TimeSpan totalDuration = TimeSpan.Zero;
        int tailLength = 6;
        var chunkSamples = SampleHelper.GetChunkSampleFormFile(files, ref totalDuration, chunkSize: 3200, tailLength: tailLength);
        if (!chunkSamples.HasValue) throw new Exception("Failed to read online audio files");
        string recognizerType = "online";
        string outputFormat = "text";
        string modelAccuracy = "int8";
        int threads = 2;
        string methodType = "chunk"; // one/batch/chunk
        var recognizer = new OnlineAliParaformerAsrRecognizer();
        SetOnlineRecognizerCallbackForResult(recognizer, recognizerType, outputFormat);
        if (methodType == "chunk")
        {
            foreach (var streamSamples in chunkSamples.Value.samplesList)
            {
                foreach (var sampleChunk in streamSamples)
                {
                    var chunk = new List<List<float[]>> { new List<float[]> { sampleChunk } };
                    await recognizer.RecognizeAsync(
                        chunk, _modelBase, _modelName, modelAccuracy, methodType, threads);
                }
            }
        }
        else
        {
            await recognizer.RecognizeAsync(
                        chunkSamples.Value.samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
        }
    }

    //BaiduTransAPI baiduTransAPI = new BaiduTransAPI();
    int i = 1;
    private async void AppendAsrResults(string str, bool isAppend = true)
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
                            //EditorAsrResults.CursorPosition = (int)EditorAsrResults.Height;
                        }
                        ));

        //TaskFactory taskFactory = new TaskFactory();
        //await taskFactory.StartNew(async () =>
        //{
        //    AsrResults2.Dispatcher.Dispatch(
        //            new Action(
        //                delegate
        //                {
        //                    int x = str.Split("\n").Length;
        //                    if (x > i)
        //                    {
        //                        string transStr = BaiduTransAPI.Trans(str.Split("\n")[x - 2]) + "\n";
        //                        if (isAppend)
        //                        {
        //                            AsrResults2.Text += transStr;
        //                        }
        //                        else
        //                        {
        //                            AsrResults2.Text += transStr;
        //                        }
        //                        i++;
        //                    }
        //                }
        //                ));
        //});
    }
    private void ClearAsrResults()
    {
        AsrResults.Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            AsrResults.Text = "";
                        }
                        ));
    }
    private void AppendAsrLogs(string str, bool isAppend = true)
    {
        //_asrLogs.Append(str);
        BtnShowAsrLogs.Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            if (BtnShowAsrLogs.IsEnabled == false)
                            {
                                BtnShowAsrLogs.IsEnabled = true;
                            }
                        }));
    }
    private void ClearAsrLogs()
    {
        //_asrLogs.Clear();
        BtnShowAsrLogs.Dispatcher.Dispatch(
                    new Action(
                        delegate
                        {
                            if (BtnShowAsrLogs.IsEnabled == true)
                            {
                                BtnShowAsrLogs.IsEnabled = false;
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
    private void OnEditAsrResultsClicked(object sender, EventArgs e)
    {
        EditorAsrResults.Text = AsrResults.Text;
        EditorAsrResults.IsVisible = true;
        EditorAsrResults.HeightRequest = AsrResults.Height;
        AsrResults.IsVisible = false;
        BtnEditAsrResults.IsVisible = false;
        BtnEditedAsrResults.IsVisible = true;
    }

    private void OnEditedAsrResultsClicked(object sender, EventArgs e)
    {
        AsrResults.Text = EditorAsrResults.Text;
        EditorAsrResults.IsVisible = false;
        AsrResults.IsVisible = true;
        BtnEditAsrResults.IsVisible = true;
        BtnEditedAsrResults.IsVisible = false;
    }

    private void ResetComponent()
    {
        ClearAsrResults();
        ClearAsrLogs();
        AsrResults.Text = "";
        EditorAsrResults.Text = "";
        EditorAsrResults.IsVisible = false;
        AsrResults.IsVisible = true;
        BtnEditAsrResults.IsVisible = true;
        BtnEditedAsrResults.IsVisible = false;
    }

    private async void OnShowAsrLogsClicked(object sender, EventArgs e)
    {
        //if (string.IsNullOrEmpty(_asrLogs.ToString()))
        //{
        //    return;
        //}
        //await DisplayAlert("Tips", _asrLogs.ToString(), "close");
    }
    #region callback    
    private async void SetOnlineRecognizerCallbackForResult(OnlineAliParaformerAsrRecognizer recognizer, string? recognizerType = "online", string outputFormat = "text")
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
    private void SetOnlineRecognizerCallbackForCompleted(OnlineAliParaformerAsrRecognizer recognizer)
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

