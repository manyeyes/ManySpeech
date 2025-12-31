using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.WindowsForms.Sample
{
    internal class SysConf
    {
        //应用程序的专用数据存储目录路径
        private static string _appDataPath;//应用程序的安装目录或程序集所在目录
        private static string _applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        public SysConf() { }

        public static string ApplicationBase { get => _applicationBase; set => _applicationBase = value; }
        public static string AppDataPath
        {
            get
            {
                // 懒加载：首次访问时初始化路径
                if (string.IsNullOrEmpty(_appDataPath))
                {
                    InitAppDataPath();
                }
                return _appDataPath;
            }
        }

        // 初始化路径
        private static void InitAppDataPath()
        {
            // 获取系统为当前用户分配的「本地应用数据目录」
            string basePath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            // 创建应用专属子目录（格式：公司名\应用名，避免与其他应用冲突）
            // 可自定义目录名，建议包含公司/应用标识，防止文件混乱
            string companyName = "manyeyes"; // 替换为你的公司名/组织名
            string appName = "manyspeech.winforms.sample"; // 替换为你的应用名称
            _appDataPath = Path.Combine(basePath, companyName, appName);

            // 确保目录存在
            if (!Directory.Exists(_appDataPath))
            {
                Directory.CreateDirectory(_appDataPath);
            }
        }
    }
}
