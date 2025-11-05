using System.ComponentModel;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Input;
using System.Net.Http;

namespace PreProcessUtils
{
    /// <summary>
    /// Enumeration representing the state of a file download
    /// </summary>
    public enum DownloadState
    {
        cancelled = 0, // Cancelled
        inprogres = 1, // In Progress
        completed = 2, // Completed
        error = 3,     // Error
        existed = 4,   // Already Exists
        noexisted = 5  // Does Not Exist
    }

    /// <summary>
    /// Delegate for download status callback
    /// </summary>
    /// <param name="progress">Current download progress (0-100)</param>
    /// <param name="downloadState">Current download state</param>
    /// <param name="filename">Name of the file being downloaded</param>
    /// <param name="msg">Additional message (e.g., error details)</param>
    public delegate void DelegateDone(int progress, DownloadState downloadState, string filename, string msg = "");

    /// <summary>
    /// Helper class for managing file downloads, including anti-crawling measures and progress tracking
    /// </summary>
    public class DownloadHelper : INotifyPropertyChanged
    {
        private readonly DelegateDone _callback;
        private bool _isDownloading = false;
        private int _progress;
        private string _fileName;
        private readonly object _lockobj = new object();
        //private string _baseFolder;
        private HttpClient _httpClient;
        private CancellationTokenSource _cts; // Used to cancel the download
        private ICommand _downloadCommand;

        // For anti-crawling: List of random User-Agents
        private readonly List<string> _userAgents = new List<string>
        {
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0",
            "Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36 Edg/112.0.1722.28",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 OPR/98.0.0.0",
            "Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/109.0 Firefox/112.0",
            "Mozilla/5.0 (iPad; CPU OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1"
        };

        // For anti-crawling: List of random Referers
        private readonly List<string> _referers = new List<string>
        {
            "https://www.bing.com/",
            "https://www.yahoo.com/",
            "https://www.facebook.com/",
            "https://www.twitter.com/",
            "https://www.linkedin.com/",
            "https://www.instagram.com/",
            "https://www.pinterest.com/",
            "https://www.reddit.com/",
            "https://www.wikipedia.org/",
            "https://example.com"
        };

        /// <summary>
        /// Indicates whether a download is currently in progress
        /// </summary>
        public bool IsDownloading
        {
            get => _isDownloading;
            set
            {
                if (value != _isDownloading)
                {
                    _isDownloading = value;
                    NotifyPropertyChanged(nameof(IsDownloading));
                }
            }
        }

        /// <summary>
        /// Name of the file being downloaded
        /// </summary>
        public string FileName
        {
            get => _fileName;
            set
            {
                if (value != _fileName)
                {
                    _fileName = value;
                    NotifyPropertyChanged(nameof(FileName));
                }
            }
        }

        /// <summary>
        /// Event triggered when a property value changes
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Triggers the PropertyChanged event
        /// </summary>
        /// <param name="property">Name of the property that changed</param>
        protected void NotifyPropertyChanged(string property)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(property));
        }

        /// <summary>
        /// Default constructor for DownloadHelper
        /// </summary>
        public DownloadHelper()
        {
        }

        /// <summary>
        /// Constructor for DownloadHelper with base folder and callback configuration
        /// </summary>
        /// <param name="baseFolder">Base folder for storing downloads</param>
        /// <param name="callback">Callback for download status updates</param>
        public DownloadHelper(string baseFolder, DelegateDone callback)
        {
            _callback = callback;
        }

        /// <summary>
        /// Creates a download task: initializes folders, checks file existence, and configures download command
        /// </summary>
        /// <param name="downloadUrl">URL of the file to download</param>
        /// <param name="fileName">Name of the file to save</param>
        /// <param name="rootFolder">Root folder for the download</param>
        /// <param name="modelName">Model name (used for folder structure)</param>
        public void DownloadCreate(string downloadUrl, string fileName, string modelBase, string modelName)
        {
            lock (_lockobj)
            {
                FileName = fileName;
                var modelFolder = Path.Combine(modelBase, modelName);

                // Create folders if they don't exist
                Directory.CreateDirectory(modelBase);
                Directory.CreateDirectory(modelFolder);

                string fileFullname = Path.Combine(modelFolder, fileName);

                if (!File.Exists(fileFullname))
                {
                    // Initialize HttpClient with anti-crawling configuration
                    InitializeHttpClient();

                    // Initialize cancellation token source
                    _cts = new CancellationTokenSource();

                    // Define download command (executed asynchronously)
                    _downloadCommand = new Command(async () =>
                    {
                        if (!IsDownloading)
                        {
                            IsDownloading = true;
                            await DownloadFileAsync(new Uri(downloadUrl), fileFullname, _cts.Token);
                        }
                    });
                }
                else
                {
                    // File already exists: clear command and trigger "existed" callback
                    _downloadCommand = null;
                    _callback?.Invoke(100, DownloadState.existed, FileName);
                }
            }
        }

        /// <summary>
        /// Initializes HttpClient with anti-crawling headers (User-Agent, Referer, etc.)
        /// </summary>
        private void InitializeHttpClient()
        {
            var handler = new HttpClientHandler
            {
                CookieContainer = new CookieContainer(),
                UseProxy = false, // Do not use proxy to avoid anti-crawling detection
                Proxy = null
            };

            _httpClient = new HttpClient(handler);
            _httpClient.Timeout = Timeout.InfiniteTimeSpan; // Unlimited timeout (controlled by cancellation token)

            // Randomly set User-Agent and Referer for anti-crawling
            var random = new Random();
            string randomUserAgent = _userAgents[random.Next(_userAgents.Count)];
            string randomReferer = _referers[random.Next(_referers.Count)];

            _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(randomUserAgent);
            _httpClient.DefaultRequestHeaders.Referrer = new Uri(randomReferer);
            _httpClient.DefaultRequestHeaders.Accept.ParseAdd("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9");
            _httpClient.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.9");
        }

        /// <summary>
        /// Core method: Downloads file asynchronously using HttpClient
        /// </summary>
        /// <param name="url">URI of the file to download</param>
        /// <param name="filePath">Local path to save the downloaded file</param>
        /// <param name="cancellationToken">Token to cancel the download</param>
        private async Task DownloadFileAsync(Uri url, string filePath, CancellationToken cancellationToken)
        {
            try
            {
                // Send request (only get response headers, not content immediately)
                using (var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                {
                    response.EnsureSuccessStatusCode(); // Check HTTP status code (e.g., 403, 404)

                    long totalBytes = response.Content.Headers.ContentLength ?? 0;
#if NET472_OR_GREATER
                    using (var httpStream = await response.Content.ReadAsStreamAsync())
#endif
#if NET6_0_OR_GREATER
                    using (var httpStream = await response.Content.ReadAsStreamAsync(cancellationToken))
#endif
                    using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        byte[] buffer = new byte[65536]; // 64KB buffer for stream reading
                        int bytesRead;
                        long totalRead = 0;

                        // Loop to read stream and write to file
                        while ((bytesRead = await httpStream.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                        {
                            await fileStream.WriteAsync(buffer, 0, bytesRead, cancellationToken);
                            totalRead += bytesRead;

                            // Calculate progress and trigger callback
                            if (totalBytes > 0)
                            {
                                _progress = (int)((double)totalRead / totalBytes * 100);
                                _callback?.Invoke(_progress, DownloadState.inprogres, FileName);
                            }
                        }
                    }
                }

                // Trigger "completed" callback after successful download
                _callback?.Invoke(100, DownloadState.completed, FileName);
            }
            catch (OperationCanceledException)
            {
                // Download cancelled: delete incomplete file and trigger callback
                if (File.Exists(filePath))
                    File.Delete(filePath);
                _callback?.Invoke(_progress, DownloadState.cancelled, FileName);
            }
            catch (Exception ex)
            {
                // Download error: delete incomplete file and trigger callback with error message
                if (File.Exists(filePath))
                    File.Delete(filePath);
                _callback?.Invoke(_progress, DownloadState.error, FileName, ex.Message);
            }
            finally
            {
                // Reset download status and dispose cancellation token
                IsDownloading = false;
                _cts?.Dispose();
            }
        }

        /// <summary>
        /// Starts the configured download task
        /// </summary>
        public void DownloadStart()
        {
            if (_downloadCommand != null && _downloadCommand.CanExecute(null))
            {
                _downloadCommand.Execute(null);
            }
        }

        /// <summary>
        /// Cancels the ongoing download
        /// </summary>
        public void DownloadCancel()
        {
            if (_cts != null && !_cts.IsCancellationRequested)
            {
                _cts.Cancel(); // Trigger cancellation token
            }
        }

        /// <summary>
        /// Checks the validity of downloaded model files (existence, size, and MD5 hash)
        /// </summary>
        /// <param name="modelFiles">Dictionary of {filename: expected MD5 hash}</param>
        /// <param name="rootFolderName">Root folder of the model files</param>
        /// <param name="modelName">Model name (for folder path)</param>
        /// <returns>True if all files are valid; False otherwise</returns>
        public bool GetDownloadState(Dictionary<string, string> modelFiles, string modelBase, string modelName)
        {
            bool isAllValid = true;
            var modelFolder = Path.Combine(modelBase, modelName);
            if (!Directory.Exists(modelFolder))
            {
                isAllValid = false;
            }
            else
            {
                // Create folders if they don't exist
                //    Directory.CreateDirectory(modelBase);
                //Directory.CreateDirectory(modelFolder);

                foreach (var modelFile in modelFiles)
                {
                    bool isValid=DownloadCheck(fileName: modelFile.Key, modelBase: modelBase, modelName: modelName, md5Str: modelFile.Value);
                    isAllValid = isAllValid && isValid;
                    //string fileFullname = Path.Combine(modelFolder, modelFile.Key);
                    //if (File.Exists(fileFullname))
                    //{
                    //    FileInfo fileInfo = new FileInfo(fileFullname);
                    //    // Check if file is empty
                    //    if (fileInfo.Length == 0)
                    //    {
                    //        isAllValid = false && isAllValid;
                    //    }
                    //    else
                    //    {
                    //        if (!string.IsNullOrEmpty(modelFile.Value))
                    //        {
                    //            // Check if file MD5 matches expected value
                    //            string actualMd5 = GetMD5Hash(fileFullname);
                    //            if (!actualMd5.Equals(modelFile.Value, StringComparison.OrdinalIgnoreCase))
                    //            {
                    //                isAllValid = false && isAllValid;
                    //            }
                    //        }
                    //    }
                    //}
                    //else
                    //{
                    //    // File does not exist
                    //    isAllValid = false && isAllValid;
                    //}
                }
            }
            return isAllValid;
        }

        /// <summary>
        /// Checks a specific downloaded file (validity, MD5 match) and handles invalid files
        /// </summary>
        /// <param name="fileName">Name of the file to check</param>
        /// <param name="rootFolderName">Root folder of the file</param>
        /// <param name="modelName">Model name (for folder path)</param>
        /// <param name="md5Str">Expected MD5 hash of the file</param>
        public bool DownloadCheck(string fileName, string modelBase, string modelName, string md5Str)
        {
            bool isValid = true;
            var downloadFolder = modelBase;
            var modelFolder = Path.Combine(downloadFolder, modelName);
            if (!Directory.Exists(modelFolder))
            {
                _callback?.Invoke(0, DownloadState.noexisted, fileName);
                isValid = false;
            }
            else
            {
                // Create folders if they don't exist
                //Directory.CreateDirectory(downloadFolder);
                //Directory.CreateDirectory(modelFolder);

                string fileFullname = Path.Combine(modelFolder, fileName);

                if (File.Exists(fileFullname))
                {
                    FileInfo fileInfo = new FileInfo(fileFullname);
                    // Delete empty file and trigger "noexisted" callback
                    if (fileInfo.Length == 0)
                    {
                        File.Delete(fileFullname);
                        _callback?.Invoke(0, DownloadState.noexisted, fileName);
                        isValid = false;
                    }
                    else
                    {
                        if (!string.IsNullOrEmpty(md5Str))
                        {
                            // Check MD5 hash; delete and trigger callback if mismatch
                            string actualMd5 = GetMD5Hash(fileFullname);
                            if (!actualMd5.Equals(md5Str, StringComparison.OrdinalIgnoreCase))
                            {
                                File.Delete(fileFullname);
                                _callback?.Invoke(0, DownloadState.noexisted, fileName);
                                isValid = false;
                            }
                        }
                        else
                        {
                            _callback?.Invoke(0, DownloadState.existed, fileName);
                            isValid = true;
                        }
                    }
                }
                else
                {
                    // File does not exist: trigger "noexisted" callback
                    _callback?.Invoke(0, DownloadState.noexisted, fileName);
                    isValid = false;
                }
            }
            return isValid;
        }

        /// <summary>
        /// Calculates the MD5 hash of a file
        /// </summary>
        /// <param name="path">Full path of the file</param>
        /// <returns>Lowercase MD5 hash string of the file</returns>
        public string GetMD5Hash(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var md5 = MD5.Create())
            {
                byte[] hashBytes = md5.ComputeHash(fs);
                StringBuilder sb = new StringBuilder();
                foreach (byte b in hashBytes)
                {
                    sb.Append(b.ToString("x2")); // Convert byte to 2-digit hex string
                }
                return sb.ToString();
            }
        }
    }

    /// <summary>
    /// Simple implementation of ICommand for handling download actions
    /// </summary>
    public class Command : ICommand
    {
        private readonly Action _execute;
        private readonly Func<bool> _canExecute;

        /// <summary>
        /// Event triggered when the ability to execute the command changes
        /// </summary>
        public event EventHandler CanExecuteChanged;

        /// <summary>
        /// Constructor for Command
        /// </summary>
        /// <param name="execute">Action to perform when the command is executed</param>
        /// <param name="canExecute">Function to determine if the command can be executed</param>
        public Command(Action execute, Func<bool> canExecute = null)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute));
            _canExecute = canExecute;
        }

        /// <summary>
        /// Determines if the command can be executed
        /// </summary>
        /// <param name="parameter">Command parameter (not used here)</param>
        /// <returns>True if the command can be executed; False otherwise</returns>
        public bool CanExecute(object parameter) => _canExecute?.Invoke() ?? true;

        /// <summary>
        /// Executes the command action
        /// </summary>
        /// <param name="parameter">Command parameter (not used here)</param>
        public void Execute(object parameter) => _execute();

        /// <summary>
        /// Triggers the CanExecuteChanged event to re-evaluate command executability
        /// </summary>
        public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }
}