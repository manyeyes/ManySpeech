#if ANDROID
using ManySpeech.Maui.Sample.Platforms.Android;
#endif
using Microsoft.Extensions.Logging;
using AudioInOut.Recorder;

namespace ManySpeech.Maui.Sample
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

#if DEBUG
    		builder.Logging.AddDebug();
#endif
#if ANDROID24_0_OR_GREATER
            builder.Services.AddSingleton<AudioInOut.Base.IRecorder, AndroidAudioCaptureService>(provider => new AndroidAudioCaptureService(200));
#else
            builder.Services.AddSingleton<AudioInOut.Base.IRecorder, WindowsWaveInRecorder>(provider => new WindowsWaveInRecorder(200));   
#endif
            builder.Services.AddTransient<K2transducerOnlineAsr>();
            builder.Services.AddTransient<ParaformerOnlineAsr>(); 
            builder.Services.AddTransient<WenetOnlineAsr>();
            builder.Services.AddTransient<MoonshineOnlineAsr>();
            return builder.Build();
        }
    }
}
