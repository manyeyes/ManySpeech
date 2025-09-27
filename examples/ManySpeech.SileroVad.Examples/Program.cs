// See https://aka.ms/new-console-template for more information
// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
namespace ManySpeech.SileroVad.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;

        [STAThread]
        private static void Main()
        {
            TestOfflineVad();
            TestOnlineVad();
        }
    }
}


