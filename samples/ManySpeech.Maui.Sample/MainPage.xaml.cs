namespace ManySpeech.Maui.Sample
{
    public partial class MainPage : ContentPage
    {
        private Timer _timer;
        private DateTime _currentTime;

        public MainPage()
        {
            InitializeComponent();
            StartClock(); 
        }

        private void StartClock()
        {
            // 立即更新一次时间
            UpdateTimeAndGreeting();

            // 创建定时器，使用正确的构造函数
            _timer = new Timer(TimerCallback, null, 0, 1000);
        }

        // 定时器回调方法
        private void TimerCallback(object state)
        {
            // 在UI线程上更新UI
            MainThread.BeginInvokeOnMainThread(UpdateTimeAndGreeting);
        }

        private void UpdateTimeAndGreeting()
        {
            _currentTime = DateTime.Now;

            // 更新时间显示 (例如: 2023-09-27 09:45:30)
            string timeText = _currentTime.ToString("yyyy-MM-dd HH:mm:ss");

            // 根据时间获取问候语
            string greeting = GetGreetingByTime(_currentTime.Hour);

            // 更新Label内容
            TimeLabel.Text = $"{greeting}\n{timeText}";

            // 为屏幕阅读器添加提示
            SemanticScreenReader.Announce($"Current time is {timeText}. {greeting}");
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

        private async void OnLinkTapped(object sender, EventArgs e)
        {
            string url = "https://github.com/manyeyes/ManySpeech";
            // 检查URL是否有效
            if (Uri.IsWellFormedUriString(url, UriKind.Absolute))
            {
                // 打开默认浏览器
                await Launcher.Default.OpenAsync(new Uri(url));
            }
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            // 页面消失时停止定时器，释放资源
            if (_timer != null)
            {
                _timer.Dispose();
                _timer = null;
            }
        }
    }
}
