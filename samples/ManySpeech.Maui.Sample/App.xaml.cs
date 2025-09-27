namespace ManySpeech.Maui.Sample
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            //MainPage = new AppShell();
            MainPage = new MySplashPage();
            _ = EndSplash();
        }
        async Task EndSplash()
        {
            await Task.Delay(1000);
            MainThread.BeginInvokeOnMainThread(() =>
            {
                MainPage = new AppShell();
            });
        }
    }
}
