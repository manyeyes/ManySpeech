namespace ManySpeech.WinForms.Sample
{
    partial class OfflineAsr
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            LblTitle = new Label();
            tableLayoutPanel2 = new TableLayoutPanel();
            DownloadProgressBar = new ProgressBar();
            panel2 = new Panel();
            RTBResults = new RichTextBox();
            panel3 = new Panel();
            ModelStatusLabel = new Label();
            BtnRecognitionFiles = new Button();
            BtnRecognitionExample = new Button();
            BtnDeleteModels = new Button();
            BtnDownLoadModels = new Button();
            BtnCheckModels = new Button();
            DownloadProgressLabel = new Label();
            tableLayoutPanel2.SuspendLayout();
            panel2.SuspendLayout();
            panel3.SuspendLayout();
            SuspendLayout();
            // 
            // LblTitle
            // 
            LblTitle.AutoSize = true;
            LblTitle.Dock = DockStyle.Fill;
            LblTitle.Font = new Font("Microsoft YaHei UI", 16F);
            LblTitle.Location = new Point(3, 0);
            LblTitle.Name = "LblTitle";
            LblTitle.Size = new Size(794, 50);
            LblTitle.TabIndex = 0;
            LblTitle.Text = "...";
            // 
            // tableLayoutPanel2
            // 
            tableLayoutPanel2.ColumnCount = 1;
            tableLayoutPanel2.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            tableLayoutPanel2.Controls.Add(DownloadProgressBar, 0, 3);
            tableLayoutPanel2.Controls.Add(panel2, 0, 4);
            tableLayoutPanel2.Controls.Add(panel3, 0, 1);
            tableLayoutPanel2.Controls.Add(LblTitle, 0, 0);
            tableLayoutPanel2.Controls.Add(DownloadProgressLabel, 0, 2);
            tableLayoutPanel2.Dock = DockStyle.Fill;
            tableLayoutPanel2.Location = new Point(0, 0);
            tableLayoutPanel2.Name = "tableLayoutPanel2";
            tableLayoutPanel2.RowCount = 5;
            tableLayoutPanel2.RowStyles.Add(new RowStyle(SizeType.Absolute, 50F));
            tableLayoutPanel2.RowStyles.Add(new RowStyle(SizeType.Absolute, 56F));
            tableLayoutPanel2.RowStyles.Add(new RowStyle(SizeType.Absolute, 50F));
            tableLayoutPanel2.RowStyles.Add(new RowStyle(SizeType.Absolute, 12F));
            tableLayoutPanel2.RowStyles.Add(new RowStyle());
            tableLayoutPanel2.Size = new Size(800, 450);
            tableLayoutPanel2.TabIndex = 2;
            // 
            // DownloadProgressBar
            // 
            DownloadProgressBar.Dock = DockStyle.Fill;
            DownloadProgressBar.Location = new Point(3, 159);
            DownloadProgressBar.Name = "DownloadProgressBar";
            DownloadProgressBar.Size = new Size(794, 6);
            DownloadProgressBar.TabIndex = 2;
            // 
            // panel2
            // 
            panel2.Controls.Add(RTBResults);
            panel2.Dock = DockStyle.Fill;
            panel2.Location = new Point(3, 171);
            panel2.Name = "panel2";
            panel2.Size = new Size(794, 279);
            panel2.TabIndex = 2;
            // 
            // RTBResults
            // 
            RTBResults.Dock = DockStyle.Fill;
            RTBResults.Location = new Point(0, 0);
            RTBResults.Name = "RTBResults";
            RTBResults.Size = new Size(794, 279);
            RTBResults.TabIndex = 0;
            RTBResults.Text = "";
            // 
            // panel3
            // 
            panel3.BackColor = SystemColors.Info;
            panel3.Controls.Add(ModelStatusLabel);
            panel3.Controls.Add(BtnRecognitionFiles);
            panel3.Controls.Add(BtnRecognitionExample);
            panel3.Controls.Add(BtnDeleteModels);
            panel3.Controls.Add(BtnDownLoadModels);
            panel3.Controls.Add(BtnCheckModels);
            panel3.Dock = DockStyle.Fill;
            panel3.Location = new Point(3, 53);
            panel3.Name = "panel3";
            panel3.Size = new Size(794, 50);
            panel3.TabIndex = 1;
            // 
            // ModelStatusLabel
            // 
            ModelStatusLabel.AutoSize = true;
            ModelStatusLabel.Location = new Point(3, 10);
            ModelStatusLabel.Name = "ModelStatusLabel";
            ModelStatusLabel.Size = new Size(130, 20);
            ModelStatusLabel.TabIndex = 4;
            ModelStatusLabel.Text = "model not ready";
            // 
            // BtnRecognitionFiles
            // 
            BtnRecognitionFiles.BackColor = SystemColors.Highlight;
            BtnRecognitionFiles.ForeColor = SystemColors.HighlightText;
            BtnRecognitionFiles.Location = new Point(557, 9);
            BtnRecognitionFiles.Name = "BtnRecognitionFiles";
            BtnRecognitionFiles.Size = new Size(109, 29);
            BtnRecognitionFiles.TabIndex = 3;
            BtnRecognitionFiles.Text = "Select Files";
            BtnRecognitionFiles.UseVisualStyleBackColor = false;
            BtnRecognitionFiles.Click += OnBtnRecognitionFilesClicked;
            // 
            // BtnRecognitionExample
            // 
            BtnRecognitionExample.BackColor = SystemColors.Highlight;
            BtnRecognitionExample.ForeColor = SystemColors.HighlightText;
            BtnRecognitionExample.Location = new Point(457, 10);
            BtnRecognitionExample.Name = "BtnRecognitionExample";
            BtnRecognitionExample.Size = new Size(94, 29);
            BtnRecognitionExample.TabIndex = 2;
            BtnRecognitionExample.Text = "Example";
            BtnRecognitionExample.UseVisualStyleBackColor = false;
            BtnRecognitionExample.Click += OnBtnRecognitionExampleClicked;
            // 
            // BtnDeleteModels
            // 
            BtnDeleteModels.Location = new Point(355, 8);
            BtnDeleteModels.Name = "BtnDeleteModels";
            BtnDeleteModels.Size = new Size(94, 29);
            BtnDeleteModels.TabIndex = 1;
            BtnDeleteModels.Text = "Delete";
            BtnDeleteModels.UseVisualStyleBackColor = true;
            BtnDeleteModels.Click += BtnDeleteModels_Click;
            // 
            // BtnDownLoadModels
            // 
            BtnDownLoadModels.Location = new Point(254, 8);
            BtnDownLoadModels.Name = "BtnDownLoadModels";
            BtnDownLoadModels.Size = new Size(95, 29);
            BtnDownLoadModels.TabIndex = 0;
            BtnDownLoadModels.Text = "DownLoad";
            BtnDownLoadModels.UseVisualStyleBackColor = true;
            BtnDownLoadModels.Click += BtnDownLoadModels_Click;
            // 
            // BtnCheckModels
            // 
            BtnCheckModels.Location = new Point(154, 9);
            BtnCheckModels.Name = "BtnCheckModels";
            BtnCheckModels.Size = new Size(94, 29);
            BtnCheckModels.TabIndex = 0;
            BtnCheckModels.Text = "Check";
            BtnCheckModels.UseVisualStyleBackColor = true;
            BtnCheckModels.Click += BtnCheckModels_Click;
            // 
            // DownloadProgressLabel
            // 
            DownloadProgressLabel.AutoSize = true;
            DownloadProgressLabel.Dock = DockStyle.Fill;
            DownloadProgressLabel.Location = new Point(3, 106);
            DownloadProgressLabel.Name = "DownloadProgressLabel";
            DownloadProgressLabel.Size = new Size(794, 50);
            DownloadProgressLabel.TabIndex = 1;
            DownloadProgressLabel.Text = "...";
            // 
            // OfflineAsr
            // 
            AutoScaleDimensions = new SizeF(9F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.Window;
            ClientSize = new Size(800, 450);
            Controls.Add(tableLayoutPanel2);
            FormBorderStyle = FormBorderStyle.None;
            Name = "OfflineAsr";
            Text = "SensevoiceOfflineAsr";
            tableLayoutPanel2.ResumeLayout(false);
            tableLayoutPanel2.PerformLayout();
            panel2.ResumeLayout(false);
            panel3.ResumeLayout(false);
            panel3.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private Button BtnCheckModels;
        private Panel panel2;
        private RichTextBox RTBResults;
        private Panel panel3;
        private Button BtnDownLoadModels;
        private Label LblTitle;
        private Button BtnDeleteModels;
        private Label DownloadProgressLabel;
        private Button BtnRecognitionFiles;
        private Button BtnRecognitionExample;
        private Label ModelStatusLabel;
        private TableLayoutPanel tableLayoutPanel2;
        private ProgressBar DownloadProgressBar;
    }
}