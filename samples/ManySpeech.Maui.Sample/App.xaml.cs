namespace ManySpeech.Maui.Sample
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
            MainPage = new MySplashPage();
            _ = EndSplash();
        }
        async Task EndSplash()
        {
            try
            {
                await Task.Delay(1500);
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    MainPage = new AppShell();
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"MainPage switching failed: {ex.Message}");
                MainPage = new AppShell(); 
            }
        }
    }
}
