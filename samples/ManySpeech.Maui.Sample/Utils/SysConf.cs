using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.Maui.Sample.Utils
{
    internal class SysConf
    {
        //应用程序的专用数据存储目录路径
        private static string _appDataPath = Microsoft.Maui.Storage.FileSystem.AppDataDirectory;
        //应用程序的安装目录或程序集所在目录
        private static string _applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        public SysConf() { }

        //public static string ApplicationBase { get => _applicationBase; set => _applicationBase = value; }
        public static string AppDataPath { get => _appDataPath; set => _appDataPath = value; }
    }
}
