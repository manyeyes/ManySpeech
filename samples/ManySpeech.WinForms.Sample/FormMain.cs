using System.Xml;
using static System.Windows.Forms.DataFormats;
using ManySpeech.WinForms.Sample.Model;
using PreProcessUtils;

namespace ManySpeech.WinForms.Sample
{
    public partial class FormMain : Form
    {
        public FormMain()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // 1. Configure left panel (leftPanel) to dock on the left side with width 200
            leftPanel.Dock = DockStyle.Left;
            leftPanel.Width = 400;
            leftPanel.BackColor = Color.FromArgb(240, 240, 240); // Light gray background

            // 2. Configure TreeView (treeMenu) to fill the left panel
            treeMenu.Dock = DockStyle.Fill;
            treeMenu.HideSelection = false;

            // 3. Configure right panel (rightPanel) to fill remaining space
            rightPanel.Dock = DockStyle.Fill;
            rightPanel.BorderStyle = BorderStyle.FixedSingle;

            // Clear default nodes
            treeMenu.Nodes.Clear();

            Dictionary<string, Dictionary<string, List<string>>> models = new Dictionary<string, Dictionary<string, List<string>>>()
            {
                {
                    "AliPraformerAsr", // Model category
                    new Dictionary<string, List<string>>()
                    {
                        {
                            "offlineasr", // Non-streaming models
                            new List<string>()
                            {
                                "paraformer-large-zh-en-onnx-offline",
                                "paraformer-large-zh-en-timestamp-onnx-offline",
                                "paraformer-large-en-onnx-offline",
                                "paraformer-large-zh-yue-en-timestamp-onnx-offline-dengcunqin-20240805",
                                "paraformer-large-zh-yue-en-onnx-offline-dengcunqin-20240805",
                                "paraformer-seaco-large-zh-timestamp-onnx-offline",
                                "paraformer-large-wenetspeech-chuan-onnx-offline",
                                "paraformer-large-wenetspeech-chuan-int8-onnx-offline",
                                "sensevoice-small-onnx",
                                "sensevoice-small-int8-onnx",
                                "sensevoice-small-wenetspeech-yue-int8-onnx",
                                "sensevoice-small-split-embed-onnx"
                            }
                        },
                        {
                            "onlineasr", // Streaming models
                            new List<string>()
                            {
                                "paraformer-large-zh-en-onnx-online",
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
                            "offlineasr", // Non-streaming models
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
                            "onlineasr", // Streaming models
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
                            "offlineasr", // Non-streaming models
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
                            "offlineasr", // Non-streaming models
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
                            "onlineasr", // Streaming models
                            new List<string>()
                            {
                                "wenet-u2pp-conformer-aishell-onnx-online-20210601",
                                "wenet-u2pp-conformer-wenetspeech-onnx-online-20220506",
                                "wenet-u2pp-conformer-gigaspeech-onnx-online-20210728"
                            }
                        },
                        {
                            "offlineasr", // Non-streaming models
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
                            "offlineasr", // Non-streaming models
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

            // Add first-level menu nodes
            List<TreeNode> treeNodeList = new List<TreeNode>();
            // Tag stores the target form type
            TreeNode nodeHome = new TreeNode("Home") { Tag = new FormConfig { FormType = typeof(FormHome) } };
            treeNodeList.Add(nodeHome);

            string[] modelCategorys = models.Keys.ToArray();
            foreach (string modelCategory in modelCategorys)
            {
                var recognizerCategory = RecognizerCategory.AliParaformerAsr;
                // Match enum values with model category (case-insensitive for better compatibility)
                foreach (RecognizerCategory enumItem in Enum.GetValues(typeof(RecognizerCategory)))
                {
                    // Convert enum to string and match with modelCategory (case-insensitive)
                    string enumToString = enumItem.ToString().ToLowerInvariant();
                    if (string.Equals(enumToString, modelCategory, StringComparison.OrdinalIgnoreCase))
                    {
                        recognizerCategory = enumItem;
                        break; // Match found, exit loop
                    }
                }

                TreeNode node = new TreeNode(modelCategory) { Tag = new FormConfig { FormType = typeof(FormASR), StrParam = modelCategory } };

                // 1. Get all non-streaming models (offline)
                var selectedModels = models[modelCategory];
                List<string> allOfflineModels = selectedModels.GetValueOrDefault("offlineasr") ?? new List<string>();
                foreach (string modelName in allOfflineModels)
                {
                    node.Nodes.Add(new TreeNode(modelName)
                    {
                        Tag = new FormConfig
                        {
                            FormType = typeof(OfflineAsr),
                            RecognizerFormParam = new RecognizerFormParam
                            {
                                RecognizerCategory = recognizerCategory,
                                ModelName = modelName
                            }
                        }
                    });
                }

                treeNodeList.Add(node);
            }

            // System setting menu node
            TreeNode nodeSetting = new TreeNode("System setting") { Tag = new FormConfig { FormType = typeof(FormSetting) } };
            // About manyspeech submenu
            nodeSetting.Nodes.Add(new TreeNode("About manyspeech") { Tag = new FormConfig { FormType = typeof(FormAbout) } });
            treeNodeList.Add(nodeSetting);

            // Add all nodes to TreeView
            treeMenu.Nodes.AddRange(treeNodeList.ToArray());

            // Expand all nodes by default
            treeMenu.ExpandAll();

            // Initialization: Load Home form by default
            LoadChildForm(typeof(FormHome));
        }
        /// <summary>
        /// Loads a child form into the right-side container panel
        /// </summary>
        /// <param name="formConfig">Configuration object for the target child form (contains form type and parameters)</param>
        private void LoadChildForm(FormConfig formConfig)
        {
            if (formConfig == null || formConfig.FormType == null)
                return;

            // 1. Close all open child forms in the right-side container (prevent duplicate loading)
            foreach (Control ctrl in rightPanel.Controls)
            {
                if (ctrl is Form childForm)
                {
                    childForm.Close(); // Close the form
                    childForm.Dispose(); // Release resources (prevent memory leaks)
                }
            }
            rightPanel.Controls.Clear(); // Clear the container

            // 2. Dynamically create target child form instance via reflection
            Form newForm = null;
            try
            {
                // Case 1: Pass complex custom parameters (invoke constructor accepting RecognizerFormParam)
                if (formConfig.RecognizerFormParam != null)
                {
                    var constructor = formConfig.FormType.GetConstructor(new[] { typeof(RecognizerCategory), typeof(string) });
                    if (constructor != null)
                    {
                        newForm = constructor.Invoke(new object[] { formConfig.RecognizerFormParam.RecognizerCategory, formConfig.RecognizerFormParam.ModelName }) as Form;
                    }
                }
                // Case 2: Pass multiple parameters (invoke constructor matching the parameter types)
                else if (formConfig.MultiParams != null && formConfig.MultiParams.Length > 0)
                {
                    // Get constructor with parameter types matching MultiParams
                    Type[] paramTypes = formConfig.MultiParams.Select(p => p.GetType()).ToArray();
                    var constructor = formConfig.FormType.GetConstructor(paramTypes);
                    if (constructor != null)
                    {
                        newForm = constructor.Invoke(formConfig.MultiParams) as Form;
                    }
                }
                // Case 3: Pass single string parameter (invoke constructor accepting string)
                else if (!string.IsNullOrEmpty(formConfig.StrParam))
                {
                    var constructor = formConfig.FormType.GetConstructor(new[] { typeof(string) });
                    if (constructor != null)
                    {
                        newForm = constructor.Invoke(new object[] { formConfig.StrParam }) as Form;
                    }
                }
                // Default case: No parameters (invoke parameterless constructor)
                else
                {
                    newForm = Activator.CreateInstance(formConfig.FormType) as Form;
                }

                // 3. Load the child form (keep original logic unchanged)
                if (newForm != null)
                {
                    newForm.TopLevel = false;
                    newForm.FormBorderStyle = FormBorderStyle.None;
                    newForm.Dock = DockStyle.Fill;
                    newForm.BackColor = rightPanel.BackColor;

                    // 4. Add to the right-side container and display the form
                    rightPanel.Controls.Add(newForm);
                    newForm.Show();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to create child form: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }
        /// <summary>
        /// Loads a child form into the right-side container panel
        /// </summary>
        /// <param name="formType">Target child form type (use typeof(child form))</param>
        private void LoadChildForm(Type formType)
        {
            // 1. Close all open child forms in the right-side container (prevent duplicate loading)
            foreach (Control ctrl in rightPanel.Controls)
            {
                if (ctrl is Form childForm)
                {
                    childForm.Close(); // Close the form
                    childForm.Dispose(); // Release resources (prevent memory leaks)
                }
            }
            rightPanel.Controls.Clear(); // Clear the container

            if (Activator.CreateInstance(formType) is Form newForm)
            {
                // 3. Set child form properties (ensure seamless integration with the container)
                newForm.TopLevel = false;
                newForm.FormBorderStyle = FormBorderStyle.None;
                newForm.Dock = DockStyle.Fill;
                newForm.BackColor = rightPanel.BackColor;

                // 4. Add the form to the right-side container and display it
                rightPanel.Controls.Add(newForm);
                newForm.Show();
            }
        }

        private void treeMenu_NodeMouseClick(object sender, TreeNodeMouseClickEventArgs e)
        {
            // Get the Tag of the clicked node (stores the target form configuration information)
            FormConfig formConfig = e.Node.Tag as FormConfig;
            if (formConfig != null && formConfig.FormType != null)
            {
                LoadChildForm(formConfig);
            }
        }

        private void rightPanel_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
