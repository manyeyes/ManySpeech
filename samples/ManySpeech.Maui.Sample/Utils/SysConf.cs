using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.Maui.Sample.Utils
{
    internal class SysConf
    {
#if WINDOWS
        private static string _applicationBase = Microsoft.Maui.Storage.FileSystem.AppDataDirectory;
#else
        private static string _applicationBase = AppDomain.CurrentDomain.BaseDirectory;
#endif
        public SysConf() { }

        public static string ApplicationBase { get => _applicationBase; set => _applicationBase = value; }
    }
}
