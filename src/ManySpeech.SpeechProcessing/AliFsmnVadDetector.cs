using ManySpeech.AliFsmnVad;
using ManySpeech.AliFsmnVad.Model;

namespace ManySpeech.SpeechProcessing
{
    public partial class AliFsmnVadDetector : IDisposable
    {
        public bool _disposed = false;

        private string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        private FsmnVad? _detector;
        private FsmnVad? InitDetector(string modelName, string modelBasePath, string modelAccuracy = "int8", int threadsNum = 2)
        {
            if (_detector == null)
            {
                if (string.IsNullOrEmpty(modelBasePath) || string.IsNullOrEmpty(modelName))
                {
                    return null;
                }
                string modelFilePath = modelBasePath + "/" + modelName + "/model.int8.onnx";
                string configFilePath = modelBasePath + "/" + modelName + "/vad.json";
                string mvnFilePath = modelBasePath + "/" + modelName + "/vad.mvn";
                try
                {
                    string folderPath = Path.Combine(modelBasePath, modelName);
                    // 1. 检查文件夹是否存在
                    if (!Directory.Exists(folderPath))
                    {
                        Console.WriteLine($"Error: folder does not exist - {folderPath}");
                        return null;
                    }
                    // 2. 获取文件夹中所有文件的完整路径
                    // 可选参数：搜索模式（如"*.txt"筛选文本文件）、是否搜索子目录
                    string[] allFilePaths = Directory.GetFiles(folderPath);
                    foreach (string filePath in allFilePaths)
                    {
                        // 提取纯文件名（含扩展名）
                        string fileName = Path.GetFileName(filePath);
                        //Console.WriteLine(fileName);
                        if (fileName.StartsWith("model") || fileName.StartsWith("encoder"))
                        {
                            if (fileName.Contains("." + modelAccuracy + "."))
                            {
                                modelFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                            }
                            else
                            {
                                if (string.IsNullOrEmpty(modelFilePath))
                                {
                                    modelFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                                }
                            }
                        }
                        if (fileName.StartsWith("vad") && fileName.EndsWith(".json"))//fileName.EndsWith(".yaml") || 
                        {
                            configFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                        }
                        if (fileName.StartsWith("vad") && fileName.EndsWith(".mvn"))
                        {
                            mvnFilePath = modelBasePath + "/" + modelName + "/" + fileName;
                        }
                    }
                    TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
                    _detector = new FsmnVad(modelFilePath: modelFilePath, configFilePath: configFilePath, mvnFilePath: mvnFilePath, threadsNum: threadsNum, batchSize: 1);
                    TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
                    double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
                    Console.WriteLine("init_models_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
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
                    Console.WriteLine($"Error occurred: {ex.Message}");
                }
            }
            return _detector;
        }

        public SegmentEntity[]? OfflineDetector(List<float[]>? samples, string? modelBasePath, string modelName = "alifsmnvad-onnx", string modelAccuracy = "int8", int threadsNum = 2)
        {
            SegmentEntity[]? segments = null;
            if (string.IsNullOrEmpty(modelBasePath))
            {
                modelBasePath = applicationBase;
            }
            FsmnVad? detector = InitDetector(modelName, modelBasePath, modelAccuracy, threadsNum);
            if (detector == null)
            {
                Console.WriteLine("Init models failure!");
                return segments;
            }
            Console.WriteLine("Read meida in progress!");
            if (samples.Count == 0)
            {
                Console.WriteLine("No media is read!");
                return segments;
            }
            Console.WriteLine("Automatic speech recognition in progress!");
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            Console.WriteLine("multi sample vad results:\r\n");
            segments = detector.GetSegmentsByStep(samples);
            //if (_detector != null)
            //{
            //    _detector.Dispose();
            //    _detector = null;
            //}
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            Console.WriteLine("recognition_elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            return segments;
        }
        public void Dispose()
        {
            if (!_disposed)
            {
                _detector?.Dispose();
                _detector = default;
                _disposed = true;
            }
        }
    }
}
