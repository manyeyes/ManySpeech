namespace ManySpeech.DolphinAsr.Examples
{
    internal static partial class Program
    {
        [STAThread]
        private static void Main()
        {
            // 第一步：调整 .NET 线程池的最小线程数（关键！）
            // 确保线程池有足够的空闲线程供 ONNX Runtime 使用
            int workerThreads, completionPortThreads;
            ThreadPool.GetMinThreads(out workerThreads, out completionPortThreads);
            // 设置最小工作线程数为 16（大于 8，留有余量）
            ThreadPool.SetMinThreads(16, completionPortThreads);

            OfflinneDolphinAsrRecognizer.OfflineRecognizer();
        }
    }
}