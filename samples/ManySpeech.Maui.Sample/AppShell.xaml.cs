using AudioInOut.Base;
using ManySpeech.Maui.Sample.Model;
using System.Collections.Concurrent;

namespace ManySpeech.Maui.Sample
{
    public partial class AppShell : Shell
    {
        private Dictionary<string, Dictionary<string, List<string>>> models = new Dictionary<string, Dictionary<string, List<string>>>()
        {
            {
                "AliPraformerAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "paraformer-large-zh-en-onnx-offline",
                            "paraformer-large-zh-en-timestamp-onnx-offline",
                            "paraformer-large-en-onnx-offline",
                            "paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805",
                            "paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805",
                            "paraformer-seaco-large-zh-timestamp-onnx-offline",
                            "sensevoice-small-onnx",
                            "sensevoice-small-int8-onnx",
                            "sensevoice-small-wenetspeech-yue-int8-onnx",
                            "sensevoice-small-split-embed-onnx"
                        }
                    },
                    {
                        "online", // Streaming models
                        new List<string>()
                        {
                            "paraformer-large-zh-en-int8-onnx-online",
                            "paraformer-large-zh-yue-en-onnx-online-dengcunqin-20240208"
                        }
                    }
                }
            },
            {
                "FireRedAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "fireredasr-aed-large-zh-en-onnx-offline-20250124"
                        }
                    }
                }
            },
            {
                "K2TransducerAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "online", // Streaming models
                        new List<string>()
                        {
                            "k2transducer-lstm-en-onnx-online-csukuangfj-20220903",
                            "k2transducer-lstm-zh-onnx-online-csukuangfj-20221014",
                            "k2transducer-zipformer-en-onnx-online-weijizhuang-20221202",
                            "k2transducer-zipformer-en-onnx-online-zengwei-20230517",
                            "k2transducer-zipformer-multi-zh-hans-onnx-online-20231212",
                            "k2transducer-zipformer-ko-onnx-online-johnbamma-20240612",
                            "k2transducer-zipformer-ctc-small-zh-onnx-online-20250401",
                            "k2transducer-zipformer-large-zh-onnx-online-yuekai-20250630",
                            "k2transducer-zipformer-xlarge-zh-onnx-online-yuekai-20250630",
                            "k2transducer-zipformer-ctc-large-zh-onnx-online-yuekai-20250630",
                            "k2transducer-zipformer-ctc-xlarge-zh-onnx-online-yuekai-20250630"
                        }
                    },
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "k2transducer-conformer-en-onnx-offline-csukuangfj-20220513",
                            "k2transducer-conformer-zh-onnx-offline-luomingshuang-20220727",
                            "k2transducer-zipformer-en-onnx-offline-yfyeung-20230417",
                            "k2transducer-zipformer-large-en-onnx-offline-zengwei-20230516",
                            "k2transducer-zipformer-small-en-onnx-offline-zengwei-20230516",
                            "k2transducer-zipformer-zh-onnx-offline-wenetspeech-20230615",
                            "k2transducer-zipformer-zh-onnx-offline-multi-zh-hans-20230902",
                            "k2transducer-zipformer-zh-en-onnx-offline-20231122",
                            "k2transducer-zipformer-cantonese-onnx-offline-20240313",
                            "k2transducer-zipformer-th-onnx-offline-yfyeung-20240620",
                            "k2transducer-zipformer-ja-onnx-offline-reazonspeech-20240801",
                            "k2transducer-zipformer-ru-onnx-offline-20240918",
                            "k2transducer-zipformer-vi-onnx-offline-20250420",
                            "k2transducer-zipformer-ctc-zh-onnx-offline-20250703",
                            "k2transducer-zipformer-ctc-small-zh-onnx-offline-20250716"
                        }
                    }
                }
            },
            {
                "MoonshineAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "moonshine-base-en-onnx",
                            "moonshine-tiny-en-onnx"
                        }
                    }
                }
            },
            {
                "WenetAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "online", // Streaming models
                        new List<string>()
                        {
                            "wenet-u2pp-conformer-aishell-onnx-online-20210601",
                            "wenet-u2pp-conformer-wenetspeech-onnx-online-20220506",
                            "wenet-u2pp-conformer-gigaspeech-onnx-online-20210728"
                        }
                    },
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "wenet-u2pp-conformer-aishell-onnx-offline-20210601",
                            "wenet-u2pp-conformer-wenetspeech-onnx-offline-20220506",
                            "wenet-u2pp-conformer-gigaspeech-onnx-offline-20210728"
                        }
                    }
                }
            },
            {
                "WhisperAsr", // Model category
                new Dictionary<string, List<string>>()
                {
                    {
                        "offline", // Non-streaming models
                        new List<string>()
                        {
                            "whisper-tiny-onnx",
                            "whisper-tiny-en-onnx",
                            "whisper-base-onnx",
                            "whisper-base-en-onnx",
                            "whisper-small-onnx",
                            "whisper-small-en-onnx",
                            "whisper-small-cantonese-onnx",
                            "whisper-medium-onnx",
                            "whisper-medium-en-onnx",
                            "whisper-large-v1-onnx",
                            "whisper-large-v2-onnx",
                            "whisper-large-v3-onnx",
                            "whisper-large-v3-turbo-onnx",
                            "whisper-large-v3-turbo-zh-onnx",
                            "distil-whisper-small-en-onnx",
                            "distil-whisper-medium-en-onnx",
                            "distil-whisper-large-v2-en-onnx",
                            "distil-whisper-large-v3-en-onnx",
                            "distil-whipser-large-v3.5-en-onnx",
                            "distil-whisper-large-v2-multi-hans-onnx",
                            "distil-whisper-small-cantonese-onnx-alvanlii-20240404"
                        }
                    }
                }
            }
        };
        private int _flyoutId = 1000; // Unique identifier to avoid route conflicts
        private readonly ConcurrentDictionary<string, bool> _navigatingFlags = new();
        private readonly IServiceProvider _recorderProvider = MauiProgram.CreateMauiApp().Services;

        public AppShell()
        {
            InitializeComponent();
            Routing.RegisterRoute(nameof(MainPage), typeof(MainPage));
            Loaded += AppShell_Loaded;
        }

        // Initialize and dynamically add menus
        private void AppShell_Loaded(object? sender, EventArgs e)
        {
            string[] modelCategorys = models.Keys.ToArray();
            foreach (string modelCategory in modelCategorys)
            {
                var recognizerCategory = RecognizerCategory.AliParaformerAsr;
                foreach (RecognizerCategory enumItem in Enum.GetValues(typeof(RecognizerCategory)))
                {
                    string enumToString = enumItem.ToString().ToLowerInvariant();
                    if (string.Equals(enumToString, modelCategory, StringComparison.OrdinalIgnoreCase))
                    {
                        recognizerCategory = enumItem;
                        break;
                    }
                }
                // 0. Get all models
                var alipraformerModels = models[modelCategory];
                var asrFlyoutItem = new FlyoutItem
                {
                    Title = modelCategory, // Group name (required, displayed as top-level menu item)
                    Icon = "cubes2.png", // Group icon
                    FlyoutDisplayOptions = FlyoutDisplayOptions.AsSingleItem // Merge as a group
                };
                // 1. Get all offline models (non-streaming)
                List<string> allOfflineModels = alipraformerModels.GetValueOrDefault("offline") ?? new List<string>();
                // OfflineAsr
                foreach (string modelName in allOfflineModels)
                {
                    AddDynamicFlyoutItem(
                        pageType: typeof(OfflineAsr),
                        flyoutTitle: modelName,
                        flyoutIcon: "",
                        pageArgs: new object[] { recognizerCategory, modelName },
                        asrFlyoutItem
                    );
                }
                // 2. Get all online models (streaming)
                List<string> allOnlineModels = alipraformerModels.GetValueOrDefault("online") ?? new List<string>();
                foreach (string modelName in allOnlineModels)
                {
                    AddDynamicFlyoutItem(
                        pageType: typeof(OnlineAsr),
                        flyoutTitle: modelName,
                        flyoutIcon: "",
                        pageArgs: new object[] { _recorderProvider.GetService(typeof(IRecorder)), recognizerCategory, modelName },
                        asrFlyoutItem
                    );
                }
                AppShell.Current.Items.Add(asrFlyoutItem);

            }
        }

        #region Dynamically add via page type + constructor parameters
        /// <summary>
        /// Dynamically add FlyoutItem (supports any Page type + constructor parameter passing)
        /// </summary>
        /// <param name="pageType">Target page type (must inherit from ContentPage)</param>
        /// <param name="flyoutTitle">Flyout menu title</param>
        /// <param name="flyoutIcon">Flyout icon resource name</param>
        /// <param name="pageArgs">Page constructor parameters (consistent with the order of page constructor parameters)</param>
        /// <param name="flyoutItem">Parent FlyoutItem to add to</param>
        public void AddDynamicFlyoutItem(
            Type pageType,
            string flyoutTitle,
            string flyoutIcon,
            object[] pageArgs,
            FlyoutItem flyoutItem)
        {
            // Validate page type (must be a subclass of ContentPage)
            if (!typeof(ContentPage).IsAssignableFrom(pageType))
            {
                throw new ArgumentException($"Type {pageType.Name} is not a subclass of ContentPage");
            }

            // Reflectively create page instance (matches constructor parameters)
            var pageInstance = Activator.CreateInstance(pageType, pageArgs) as ContentPage;
            if (pageInstance == null)
            {
                throw new InvalidOperationException($"Failed to create instance of {pageType.Name}, please check if the constructor parameters match");
            }

            // Generate unique route (page type name + auto-increment ID)
            _flyoutId++;
            string shellContentRoute = $"{pageType.Name}_Route_{_flyoutId}";

            // Create ShellContent and add to FlyoutItem
            var shellContent = new ShellContent
            {
                Content = pageInstance,
                Route = shellContentRoute,
                Title = flyoutTitle,
            };

            flyoutItem.Items.Add(shellContent);
            Items.Add(flyoutItem);

            // Register route (optional, supports navigation via route)
            Routing.RegisterRoute(shellContentRoute, pageType);
        }
        #endregion
    }
}