using ManySpeech.SpeechProcessing;
using ManySpeech.SpeechProcessing.ASR.Base;
using ManySpeech.WindowsForms.Sample.Model;
using PreProcessUtils;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ManySpeech.WindowsForms.Sample
{
    public partial class OfflineAsr : Form
    {
        private string _modelBase = Path.Combine(SysConf.AppDataPath, "AllModels");
        // 如何使用其他模型
        // 1.打开 https://modelscope.cn/profile/manyeyes?tab=model 页面
        // 2.搜索 sensevoice, paraformer offline onnx 离线模型（非流式模型）
        // 3.设置 _modelName 值，_modelName = [模型名称]
        private string _modelName = "sensevoice-small-int8-onnx";
        // 如需强制先行检查文件，可填_modelFiles <文件名, hash>
        // hash为空时，仅判断文件是否存在
        private Dictionary<string, string> _modelFiles = new Dictionary<string, string>()
        {
            //{"model.int8.onnx",""},
            //{"am.mvn","" },
            //{"asr.json","" },
            //{"tokens.txt","" }
        };
        private IRecognizer _recognizer;
        private RecognizerCategory _recognizerCategory = RecognizerCategory.AliParaformerAsr;
        public OfflineAsr(RecognizerCategory recognizerCategory, string modelName)
        {
            InitializeComponent();
            // 强制设置子窗体属性（避免设计时遗漏）
            this.FormBorderStyle = FormBorderStyle.None; // 无边框
            this.TopLevel = false; // 非顶级窗体
            this.Dock = DockStyle.Fill; // 填充父容器（右侧面板）
            this.BackColor = Color.White;
            this.Shown += Form_Shown;
            _recognizerCategory = recognizerCategory;
            _modelName = modelName;
            LblTitle.Text = _modelName;
        }

        private void Form_Shown(object sender, EventArgs e)
        {
            CheckModels();
        }

        #region event
        private async void BtnCheckModels_Click(object sender, EventArgs e)
        {
            BtnCheckModels.Enabled = false;
            TaskFactory taskFactory = new TaskFactory();
            await taskFactory.StartNew(async () =>
            {
                CheckModels();
            });
            BtnCheckModels.Enabled = true;
        }

        private async void BtnDownLoadModels_Click(object sender, EventArgs e)
        {
            BtnDownLoadModels.Enabled = false;
            DownloadProgressBar.Value = 0;
            DownloadProgressLabel.Text = "";
            TaskFactory taskFactory = new TaskFactory();
            await taskFactory.StartNew(async () =>
            {
                DownloadModels();
            });
            BtnDownLoadModels.Enabled = true;
        }

        private async void BtnDeleteModels_Click(object sender, EventArgs e)
        {
            BtnDeleteModels.Enabled = false;
            TaskFactory taskFactory = new TaskFactory();
            await taskFactory.StartNew(() =>
            {
                DeleteModels();
            });
            BtnDeleteModels.Enabled = true;
        }
        #endregion

        private async void DownloadModels()
        {
            this.Invoke(new Action(delegate
            {
                DownloadProgressLabel.Visible = false;
                DownloadProgressLabel.Text = "";
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
            this.Invoke(new Action(async delegate
            {
                ModelStatusLabel.Visible = true;
                bool state = downloadHelper.GetDownloadState(_modelFiles, _modelBase, _modelName);
                if (state)
                {
                    ModelStatusLabel.Text = "model is ready";
                    DownloadProgressLabel.Visible = false;
                    DownloadProgressLabel.Text = "";
                }
                else
                {
                    ModelStatusLabel.Text = "model not ready";
                    DownloadProgressLabel.Visible = false;
                    DownloadProgressLabel.Text = "";
                    DownloadProgressLabel.Visible = true;
                    bool isDownload = await ShowConfirmDialog();
                    if (isDownload)
                    {
                        DownloadModels();
                    }
                }
            }));

        }
        private Task<bool> ShowConfirmDialog()
        {
            // 定义弹窗参数
            string caption = "Question?";
            string message = "Missing model, will it be automatically downloaded?";

            // 情况1：当前在 UI 线程 → 直接显示弹窗并返回结果
            if (!this.InvokeRequired)
            {
                DialogResult result = MessageBox.Show(
                    owner: this,          // 置顶当前应用窗口
                    text: message,        // 弹窗内容
                    caption: caption,     // 弹窗标题
                    buttons: MessageBoxButtons.YesNo,
                    icon: MessageBoxIcon.Question // 问号图标，符合「询问」场景
                );
                // 转换结果：Yes→true，No→false
                return Task.FromResult(result == DialogResult.Yes);
            }
            // 情况2：当前在非 UI 线程 → 调度到 UI 线程执行，返回 Task<bool> 支持 await
            else
            {
                // 用 TaskCompletionSource 包装同步结果，转为异步 Task
                var tcs = new TaskCompletionSource<bool>();
                this.Invoke(new Action(() =>
                {
                    DialogResult result = MessageBox.Show(this, message, caption, MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                    tcs.SetResult(result == DialogResult.Yes); // 完成 Task 并返回结果
                }));
                return tcs.Task;
            }
        }

        private void DownloadDisplay(int progress, DownloadState downloadState, string filename, string msg = "")
        {
            this.Invoke(new Action(delegate
            {
                if (progress == 0 && downloadState == DownloadState.inprogres)
                {

                    DownloadProgressLabel.Visible = true;
                    DownloadProgressLabel.Text = msg;
                }
                else
                {
                    switch (downloadState)
                    {
                        case DownloadState.inprogres:
                            DownloadProgressBar.Value = progress;
                            DownloadProgressLabel.Text = $"File: {filename}, downloading, progress: {progress}%\n";
                            break;
                        case DownloadState.cancelled:
                            DownloadProgressBar.Value = progress;
                            DownloadProgressLabel.Text = $"File: {filename}, download cancelled\n";
                            break;
                        case DownloadState.error:
                            DownloadProgressBar.Value = progress;
                            DownloadProgressLabel.Text = $"File: {filename}, download failed: {msg}\n";
                            break;
                        case DownloadState.completed:
                            DownloadProgressBar.Value = progress;
                            DownloadProgressLabel.Text = $"File: {filename}, download completed\n";
                            DownloadProgressLabel.Text += $"File: {filename}, download completed\n";
                            break;
                        case DownloadState.existed:
                            DownloadProgressBar.Value = progress;
                            DownloadProgressLabel.Text += $"File: {filename}, already exists\n";
                            break;
                        case DownloadState.noexisted:
                            DownloadProgressLabel.Visible = false;
                            DownloadProgressLabel.Text += $"File: {filename}, does not exist\n";
                            break;
                    }
                }
            }));
        }

        private async void OnBtnRecognitionExampleClicked(object sender, EventArgs e)
        {
            BtnRecognitionExample.Enabled = false;
            TaskFactory taskFactory = new TaskFactory();
            await taskFactory.StartNew(async () =>
            {
                await RecognizerFilesByOffline();
            });
            BtnRecognitionExample.Enabled = true;
        }

        private async void OnBtnRecognitionFilesClicked(object sender, EventArgs e)
        {
            // 1. 创建 WinForms 文件选择对话框
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                // 2. 配置文件选择规则
                openFileDialog.Title = "Please select a comic file";
                // 文件筛选器：格式为「描述|扩展名1;扩展名2」，这里匹配原代码的 .wav 和 .mp3
                openFileDialog.Filter = "音频文件 (*.wav;*.mp3)|*.wav;*.mp3|所有文件 (*.*)|*.*";
                openFileDialog.FilterIndex = 1; // 默认选中第一个筛选器（音频文件）
                openFileDialog.RestoreDirectory = true; // 打开后恢复上次选择的目录
                openFileDialog.Multiselect = false; // 仅允许选择单个文件（如需多选，设为 true）


                // 3. 显示文件选择对话框（模态窗口，用户操作后返回）
                DialogResult dialogResult = openFileDialog.ShowDialog();

                // 4. 处理用户选择结果（点击「确定」则继续）
                if (dialogResult == DialogResult.OK)
                {
                    // 获取选择的文件完整路径
                    string fullpath = openFileDialog.FileName;
                    List<string> fullpaths = new List<string> { fullpath };

                    // 5. 执行识别方法
                    bool status = await RecognizerFilesByOffline(fullpaths);
                    if (status)
                    {
                        // 识别完成后更新 UI（需通过 Invoke 调度到 UI 线程）
                        this.Invoke((Action)(() =>
                        {
                            MessageBox.Show("文件识别完成！", "提示");
                        }));
                    }
                }
            }
        }
        private async void OnBtnRecognitionClearClicked(object sender, EventArgs e)
        {
            ClearResults();
        }

        public async Task<bool> RecognizerFilesByOffline(List<string> fullpaths = null)
        {
            bool execStatus = false;
            try
            {
                string[] files = !fullpaths?.Any() ?? true ? SampleHelper.GetPaths(_modelBase, _modelName) : fullpaths.ToArray();
                if (files.Length == 0)
                {
                    ShowResults("No input files found");
                    return false;
                }
                string modelAccuracy = "int8";
                string methodType = "one";// 文件识别 -method one/batch
                int threads = 2;
                if (_recognizer == null)
                {
                    switch (_recognizerCategory)
                    {
                        case RecognizerCategory.AliParaformerAsr:
                            _recognizer = new OfflineAliParaformerAsrRecognizer();
                            break;
                        case RecognizerCategory.FireRedAsr:
                            _recognizer = new OfflineFireRedAsrRecognizer();
                            break;
                        case RecognizerCategory.K2TransducerAsr:
                            _recognizer = new OfflineK2TransducerAsrRecognizer();
                            break;
                        case RecognizerCategory.MoonshineAsr:
                            _recognizer = new OfflineMoonshineAsrRecognizer();
                            break;
                        case RecognizerCategory.WhisperAsr:
                            _recognizer = new OfflineWhisperAsrRecognizer();
                            break;
                        case RecognizerCategory.WenetAsr:
                            _recognizer = new OfflineWenetAsrRecognizer();
                            break;
                        default:
                            _recognizer = new OfflineAliParaformerAsrRecognizer();
                            break;
                    }
                    SetOfflineRecognizerCallbackForResult(_recognizer, "offline");
                    SetOfflineRecognizerCallbackForCompleted(_recognizer);
                }
                if (_recognizer == null) { return false; }
                ShowResults("Speech recognition in progress, please wait ...");
                TimeSpan totalDuration = TimeSpan.Zero;
                int tailLength = 6;
                var samples = SampleHelper.GetSampleFormFile(files, ref totalDuration);
                if (!samples.HasValue)
                {
                    ShowResults("Failed to read audio files");
                    return false;
                }
                else
                {
                    if (samples.Value.sampleList.Count == 0)
                    {
                        ShowResults("No media file is read!");
                        return false;
                    }
                    var samplesList = new List<List<float[]>>();
                    samplesList = samples.Value.sampleList.Select(x => new List<float[]>() { x }).ToList();
                    await _recognizer.RecognizeAsync(
                               samplesList, _modelBase, _modelName, modelAccuracy, methodType, threads);
                    execStatus = true;
                }
            }
            catch (Exception ex)
            {
                ShowTips(ex.Message);
                execStatus = false;
            }
            return execStatus;
        }
        private void ShowResults(string str, bool isAppend = true)
        {
            this.Invoke(new Action(async delegate
            {
                if (isAppend)
                {
                    RTBResults.Text += str + "\n";
                }
                else
                {
                    RTBResults.Text = str + "\n";
                }
                await Task.Delay(100);
            }
                            ));
        }
        private void ClearResults()
        {
            this.Invoke(new Action(delegate
            {
                RTBResults.Text = "";
            }));
        }
        private void ShowTips(string str)
        {
            // 确保在 UI 线程执行弹窗（非 UI 线程调用时自动调度）
            if (this.InvokeRequired)
            {
                // 非 UI 线程：通过 Invoke 切换到 UI 线程执行
                this.Invoke(new Action(() => ShowTips(str)));
            }
            else
            {
                // UI 线程：直接显示 MessageBox（同步模态窗口，无需异步）
                MessageBox.Show(
                    owner: this,          // 父窗口（让弹窗置顶当前应用）
                    text: str,            // 提示内容（对应原 str）
                    caption: "Tips",      // 弹窗标题（对应原 "Tips"）
                    buttons: MessageBoxButtons.OK,  // 按钮（对应原 "close" 单个按钮）
                    icon: MessageBoxIcon.Information // 信息图标（可选，更美观）
                );
            }
        }

        private void OnEditResultsClicked(object sender, EventArgs e)
        {
            RTBResults.Visible = false;
        }

        private void OnEditedResultsClicked(object sender, EventArgs e)
        {
            RTBResults.Visible = true;
        }

        #region callback
        private void SetOfflineRecognizerCallbackForResult(IRecognizer recognizer, string recognizerType)
        {
            int i = 0;
            recognizer.ResetRecognitionResultHandlers();
            recognizer.OnRecognitionResult += async result =>
            {
                string text = AEDEmojiHelper.ReplaceTagsWithEmojis(result.Text.Replace("> ", ">"));
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

        private void SetOfflineRecognizerCallbackForCompleted(IRecognizer recognizer)
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
}