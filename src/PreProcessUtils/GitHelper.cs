using LibGit2Sharp;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace PreProcessUtils
{
    public class GitHelper
    {
        private readonly DelegateDone _callback;
        // The host URL for downloading
        private string _downloadHost = "https://www.modelscope.cn/models";
        // The host URL for the repository
        private string _repoHost = "https://www.modelscope.cn/manyeyes";

        public GitHelper() { }

        public GitHelper(DelegateDone callback)
        {
            _callback = callback;
        }

        // Property to track the progress, records the progress at the last output
        private int CurrentProgress { get; set; } = -1;

        public string DownloadHost
        {
            get => _downloadHost;
            set => _downloadHost = value;
        }

        public string RepoHost
        {
            get => _repoHost;
            set => _repoHost = value;
        }

        /// <summary>
        /// Check if a directory is empty.
        /// </summary>
        /// <param name="path">The path of the directory.</param>
        /// <returns>A tuple containing a boolean indicating if it's empty and a message.</returns>
        (bool isEmpty, string message) IsDirectoryEmpty(string path)
        {
            try
            {
                // Check if the directory exists first
                if (!Directory.Exists(path))
                {
                    return (false, $"Directory does not exist: {path}");
                }

                // Get all files and subdirectories in the directory (non-recursive)
                // If the array length is 0, the directory is empty
                string[] entries = Directory.GetFileSystemEntries(path);

                if (entries.Length == 0)
                {
                    return (true, $"Directory is empty: {path}");
                }
                else
                {
                    return (false, $"Directory is not empty: {path} (contains {entries.Length} items)");
                }
            }
            catch (UnauthorizedAccessException)
            {
                return (false, $"No permission to access the directory: {path}");
            }
            catch (PathTooLongException)
            {
                return (false, $"Path is too long: {path}");
            }
            catch (IOException ex)
            {
                return (false, $"IO error: {ex.Message}");
            }
        }

        public async Task ProcessCloneModel(string baseFolder, string modelName)
        {
            if (string.IsNullOrEmpty(baseFolder) || string.IsNullOrEmpty(modelName))
            {
                return;
            }

            string repoUrl = "https://www.modelscope.cn/manyeyes/" + modelName + ".git";
            string localPath = Path.Combine(baseFolder, modelName);

            if (Directory.Exists(localPath))
            {
                DownloadDisplay(msg: $"'{localPath}' exists");
                var r = IsDirectoryEmpty(localPath);
                if (r.isEmpty)
                {
                    DownloadDisplay(msg: r.message);
                    DownloadDisplay(msg: $@"delete '{localPath}'");
                    // Perform the delete operation
                    // When recursive is true: delete the folder and all its sub-files and sub-directories
                    // When recursive is false: only delete empty folders (throw an exception if not empty)
                    Directory.Delete(path: localPath, recursive: true);
                }
                else
                {
                    return;
                }
            }

            try
            {
                var fetchOptions = new FetchOptions()
                {
                    OnProgress = progress =>
                    {
                        //Console.Write("\rremote: {0,-50}", progress);
                        DownloadDisplay(msg: string.Format("\rremote: {0,-50}", progress));
                        return true;
                    }
                };

                // Clone options for cloning the repository
                var cloneOptions = new CloneOptions(fetchOptions)
                {
                    Checkout = true
                };

                DownloadDisplay(msg: $"Cloning into '{modelName}'...");

                Repository.Clone(repoUrl, localPath, cloneOptions);

                // Use the git-lfs command line to pull LFS files
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "git-lfs",
                        Arguments = "pull",
                        WorkingDirectory = localPath,
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                try
                {
                    process.Start();
                    DownloadDisplay(msg: "Cloning in progress, please wait...");
                    // Wait for the process to exit
                    process.WaitForExit();

                    if (process.ExitCode != 0)
                    {
                        await DownloadModels(localPath, baseFolder, modelName);
                    }
                }
                catch
                {
                    await DownloadModels(localPath, baseFolder, modelName);
                }

                DownloadDisplay(msg: "Repository cloned successfully!");
            }
            catch (Exception ex)
            {
                DownloadDisplay(msg: $"Cloning failed: {ex.Message}");                
            }
        }
        public static double? ExtractNumberToDouble(string input)
        {
            if (string.IsNullOrEmpty(input))
                return null;

            string pattern = @"[-+]?\d+(\.\d+)?";
            Match match = Regex.Match(input, pattern);

            // 转换成功则返回数值，失败返回 null
            if (match.Success && double.TryParse(match.Value, out double result))
                return result;

            return null;
        }

        public async Task DeleteModels(string baseFolder, string modelName)
        {
            string localPath = Path.Combine(baseFolder, modelName);

            if (Directory.Exists(localPath))
            {
                // 检查并删除 index.lock 文件（常见的锁定文件）
                string[] targetExtensions = { ".idx", ".pack" };
                // 获取所有文件并筛选出符合条件的
                string[] mediaFilePaths = Directory.GetFiles(
                        path: localPath,
                        searchPattern: "*.*",  // 先获取所有文件
                        searchOption: SearchOption.AllDirectories
                    )
                    .Where(file => targetExtensions.Contains(
                        Path.GetExtension(file).ToLowerInvariant()  // 不区分大小写匹配
                    ))
                    .ToArray();
                foreach (var file in mediaFilePaths)
                {
                    if (File.Exists(file))
                    {
                        File.SetAttributes(file, FileAttributes.Normal);
                        File.Delete(file);
                    }
                }
                DownloadDisplay(msg: $"开始删除目录：'{localPath}'");
                try
                {
                    await Task.Run(() =>
                    {
                        Directory.Delete(localPath, recursive: true);
                    });
                    DownloadDisplay(msg: $"目录删除完成：'{localPath}'");
                }
                catch (IOException ex)
                {
                    DownloadDisplay(msg: $"删除失败（文件被占用）：{ex.Message}");
                    // 可根据需求添加重试逻辑或错误处理
                }
                catch (UnauthorizedAccessException ex)
                {
                    DownloadDisplay(msg: $"删除失败（权限不足）：{ex.Message}");
                }
            }
            else
            {
                DownloadDisplay(msg: $"目录不存在，无需删除：'{localPath}'");
            }
        }
        private async Task DownloadModels(string localPath, string baseFolder, string modelName)
        {
            try
            {
                // 1. Find LFS pointer files (execute in a Task)
                List<string> fileNames = await Task.Run(() => FindLfsPointers(localPath));
                DownloadDisplay(msg: $"Found {fileNames.Count} LFS pointer files, starting download...");

                // 2. Download files (depends on the result of step 1, execute in a Task)
                await Task.Run(() => DownLoadFile(baseFolder, modelName, fileNames));
                DownloadDisplay(msg: "All file download tasks have been completed");
            }
            catch (Exception ex)
            {
                DownloadDisplay(msg: $"An error occurred during processing: {ex.Message}");
            }
        }

        private async Task ReadStreamAsync(StreamReader reader, Action<string> callback)
        {
            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (line != null)
                {
                    callback?.Invoke(line);
                }
            }
        }

        // Check if a blob is an LFS pointer
        public bool IsLfsPointer(Blob blob)
        {
            if (blob == null) return false;

            // Read the file content (LFS pointers are usually small and have fixed content)
            using (var stream = blob.GetContentStream())
            using (var reader = new StreamReader(stream))
            {
                string content = reader.ReadToEnd();
                // Check the characteristics of an LFS pointer
                return content.StartsWith("version https://git-lfs.github.com/spec/v1")
                    && content.Contains("oid sha256:")
                    && content.Contains("size ");
            }
        }

        // Traverse the LFS pointer files in the repository
        public List<string> FindLfsPointers(string repoPath)
        {
            List<string> fileNames = new List<string>();
            using (var repo = new Repository(repoPath))
            {
                // Get all files of the latest commit
                var latestCommit = repo.Head.Tip;
                foreach (var treeEntry in latestCommit.Tree)
                {
                    if (treeEntry.TargetType == TreeEntryTargetType.Blob)
                    {
                        var blob = (Blob)treeEntry.Target;
                        if (IsLfsPointer(blob))
                        {
                            string relativePath = treeEntry.Path;
                            string fullPath = Path.Combine(repoPath, relativePath);
                            // Try to delete the file
                            if (DeleteFile(fullPath))
                            {
                                fileNames.Add(relativePath);
                            }
                        }
                    }
                }
            }
            return fileNames;
        }

        /// <summary>
        /// Delete a file and handle possible exceptions.
        /// </summary>
        private bool DeleteFile(string fullPath)
        {
            try
            {
                if (File.Exists(fullPath))
                {
                    // Ensure the file is not in use
                    File.SetAttributes(fullPath, FileAttributes.Normal);
                    File.Delete(fullPath);
                    return !File.Exists(fullPath); // Verify if it has been deleted
                }
                DownloadDisplay(msg: $"File does not exist: {fullPath}");
                return false;
            }
            catch (IOException ex)
            {
                DownloadDisplay(msg: $"IO error: Unable to delete file {fullPath}, reason: {ex.Message}");
                return false;
            }
            catch (UnauthorizedAccessException ex)
            {
                DownloadDisplay(msg: $"Insufficient permissions: Unable to delete file {fullPath}, reason: {ex.Message}");
                return false;
            }
            catch (Exception ex)
            {
                DownloadDisplay(msg: $"Failed to delete file {fullPath}, reason: {ex.Message}");
                return false;
            }
        }

        private void DownloadDisplay(int progress = 0, DownloadState downloadState = DownloadState.inprogres, string filename = "", string msg = "")
        {
            if (_callback != null)
            {
                _callback?.Invoke(progress, downloadState, filename, msg);
            }
            else
            {
                if (progress == 0)
                {
                    if (!string.IsNullOrEmpty(msg))
                    {
                        Console.Write(msg);
                    }
                }
                else
                {
                    // Record the current progress to the property
                    int newProgress = progress;
                    switch (downloadState)
                    {
                        case DownloadState.inprogres:
                            // Print to console when progress increment >= 5 or on the first output
                            if (CurrentProgress == -1 || newProgress - CurrentProgress >= 5)
                            {
                                Console.WriteLine($"File: {filename}, downloading, progress: {newProgress}%");
                                CurrentProgress = newProgress; // Update the tracked progress
                            }
                            break;
                        case DownloadState.cancelled:
                            // Output directly for cancelled state
                            Console.WriteLine($"File: {filename}, download cancelled");
                            CurrentProgress = -1;
                            break;
                        case DownloadState.error:
                            // Output directly for error state
                            Console.WriteLine($"File: {filename}, download failed: {msg}");
                            CurrentProgress = -1;
                            break;
                        case DownloadState.completed:
                            // Output directly for completed state
                            Console.WriteLine($"File: {filename}, download completed: {msg}");
                            CurrentProgress = -1;
                            break;
                        case DownloadState.existed:
                            // Output directly for existed state
                            Console.WriteLine($"File: {filename}, already exists");
                            CurrentProgress = -1;
                            break;
                        case DownloadState.noexisted:
                            // Output directly for non-existent state
                            Console.WriteLine($"File: {filename}, does not exist");
                            break;
                    }
                }
            }
        }

        private void DownLoadFile(string baseFolder, string modelName, List<string> fileNames)
        {
            List<string> indexs = new List<string>();
            DownloadHelper downloadHelper = new DownloadHelper(baseFolder, DownloadDisplay);
            while (indexs.Count < fileNames.Count || downloadHelper.IsDownloading)
            {
                if (downloadHelper.IsDownloading)
                {
                    Task.Delay(100).Wait();
                    continue;
                }
                foreach (var fileName in fileNames)
                {
                    if (indexs.Contains(fileName))
                    {
                        continue;
                    }
                    if (downloadHelper.IsDownloading)
                    {
                        break;
                    }
                    var downloadUrl = string.Format("{0}/manyeyes/{1}/resolve/{2}/{3}", _downloadHost, modelName, "master", fileName);
                    downloadHelper.DownloadCreate(downloadUrl, fileName, baseFolder, modelName);
                    downloadHelper.DownloadStart();
                    indexs.Add(fileName);
                }
            }
        }
    }
}