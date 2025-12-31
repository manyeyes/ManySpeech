using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Timer = System.Threading.Timer;

namespace ManySpeech.WindowsForms.Sample
{
    public partial class FormHome : Form
    {
        private Timer _timer;
        private DateTime _currentTime;
        public FormHome()
        {
            InitializeComponent();
            // 设置子窗体属性
            this.FormBorderStyle = FormBorderStyle.None; // 无边框
            this.TopLevel = false;
            this.Dock = DockStyle.Fill;
            this.BackColor = Color.White;
            StartClock();
        }

        private void StartClock()
        {
            UpdateTimeAndGreeting();
            _timer = new Timer(TimerCallback, null, 0, 1000);
        }

        // 定时器回调方法
        private void TimerCallback(object state)
        {
            if (this.IsDisposed || !this.IsHandleCreated)
                return;
            BeginInvoke(new Action(UpdateTimeAndGreeting));
        }

        private void UpdateTimeAndGreeting()
        {
            if (this.IsDisposed) return;
            _currentTime = DateTime.Now;
            // 更新时间显示 (例如: 2025-09-27 09:45:30)
            string timeText = _currentTime.ToString("yyyy-MM-dd HH:mm:ss");
            // 根据时间获取问候语
            string greeting = GetGreetingByTime(_currentTime.Hour);
            // 更新Label内容
            TimeLabel.Text = $"{greeting}\n{timeText}";
        }

        private string GetGreetingByTime(int hour)
        {
            if (hour >= 5 && hour < 9)
            {
                return "Good morning! It's a great start to the day.";
            }
            else if (hour >= 9 && hour < 12)
            {
                return "Good morning! Hope you're having a productive morning.";
            }
            else if (hour >= 12 && hour < 14)
            {
                return "Good noon! Maybe it's time for lunch.";
            }
            else if (hour >= 14 && hour < 17)
            {
                return "Good afternoon! The day is going well.";
            }
            else if (hour >= 17 && hour < 19)
            {
                return "Good evening! Time to wrap up the work day.";
            }
            else if (hour >= 19 && hour < 22)
            {
                return "Good evening! Enjoy your relaxing time.";
            }
            else if (hour >= 22 && hour < 24)
            {
                return "It's getting late. Maybe you should prepare for bed soon.";
            }
            else // hour >= 0 && hour < 5
            {
                return "It's the middle of the night. You should be resting now.";
            }
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            string url = "https://github.com/manyeyes/ManySpeech";
            if (!Uri.IsWellFormedUriString(url, UriKind.Absolute))
            {
                MessageBox.Show("无效的链接地址！", "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            try
            {
                Process.Start(new ProcessStartInfo(url)
                {
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"无法打开链接：{ex.Message}", "打开失败", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void FormHome_FormClosing(object sender, FormClosingEventArgs e)
        {
            // 页面消失时停止定时器，释放资源
            if (_timer != null)
            {
                _timer.Dispose();
                _timer = null;
            }
            base.OnFormClosing(e);
        }
    }
}
