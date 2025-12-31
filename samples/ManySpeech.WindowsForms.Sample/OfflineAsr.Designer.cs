using System.Drawing;
using System.Windows.Forms;

namespace ManySpeech.WindowsForms.Sample
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
            this.LblTitle = new System.Windows.Forms.Label();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.DownloadProgressBar = new System.Windows.Forms.ProgressBar();
            this.panel2 = new System.Windows.Forms.Panel();
            this.RTBResults = new System.Windows.Forms.RichTextBox();
            this.panel3 = new System.Windows.Forms.Panel();
            this.ModelStatusLabel = new System.Windows.Forms.Label();
            this.BtnRecognitionFiles = new System.Windows.Forms.Button();
            this.BtnRecognitionExample = new System.Windows.Forms.Button();
            this.BtnDeleteModels = new System.Windows.Forms.Button();
            this.BtnDownLoadModels = new System.Windows.Forms.Button();
            this.BtnCheckModels = new System.Windows.Forms.Button();
            this.DownloadProgressLabel = new System.Windows.Forms.Label();
            this.tableLayoutPanel2.SuspendLayout();
            this.panel2.SuspendLayout();
            this.panel3.SuspendLayout();
            this.SuspendLayout();
            // 
            // LblTitle
            // 
            this.LblTitle.AutoSize = true;
            this.LblTitle.Dock = System.Windows.Forms.DockStyle.Fill;
            this.LblTitle.Font = new System.Drawing.Font("Microsoft YaHei UI", 16F);
            this.LblTitle.Location = new System.Drawing.Point(3, 0);
            this.LblTitle.Name = "LblTitle";
            this.LblTitle.Size = new System.Drawing.Size(705, 38);
            this.LblTitle.TabIndex = 0;
            this.LblTitle.Text = "...";
            // 
            // tableLayoutPanel2
            // 
            this.tableLayoutPanel2.ColumnCount = 1;
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel2.Controls.Add(this.DownloadProgressBar, 0, 3);
            this.tableLayoutPanel2.Controls.Add(this.panel2, 0, 4);
            this.tableLayoutPanel2.Controls.Add(this.panel3, 0, 1);
            this.tableLayoutPanel2.Controls.Add(this.LblTitle, 0, 0);
            this.tableLayoutPanel2.Controls.Add(this.DownloadProgressLabel, 0, 2);
            this.tableLayoutPanel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanel2.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            this.tableLayoutPanel2.RowCount = 5;
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 38F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 42F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 38F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 9F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle());
            this.tableLayoutPanel2.Size = new System.Drawing.Size(711, 338);
            this.tableLayoutPanel2.TabIndex = 2;
            // 
            // DownloadProgressBar
            // 
            this.DownloadProgressBar.Dock = System.Windows.Forms.DockStyle.Fill;
            this.DownloadProgressBar.Location = new System.Drawing.Point(3, 120);
            this.DownloadProgressBar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DownloadProgressBar.Name = "DownloadProgressBar";
            this.DownloadProgressBar.Size = new System.Drawing.Size(705, 5);
            this.DownloadProgressBar.TabIndex = 2;
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.RTBResults);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel2.Location = new System.Drawing.Point(3, 129);
            this.panel2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(705, 209);
            this.panel2.TabIndex = 2;
            // 
            // RTBResults
            // 
            this.RTBResults.Dock = System.Windows.Forms.DockStyle.Fill;
            this.RTBResults.Location = new System.Drawing.Point(0, 0);
            this.RTBResults.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RTBResults.Name = "RTBResults";
            this.RTBResults.Size = new System.Drawing.Size(705, 209);
            this.RTBResults.TabIndex = 0;
            this.RTBResults.Text = "";
            // 
            // panel3
            // 
            this.panel3.BackColor = System.Drawing.SystemColors.Info;
            this.panel3.Controls.Add(this.ModelStatusLabel);
            this.panel3.Controls.Add(this.BtnRecognitionFiles);
            this.panel3.Controls.Add(this.BtnRecognitionExample);
            this.panel3.Controls.Add(this.BtnDeleteModels);
            this.panel3.Controls.Add(this.BtnDownLoadModels);
            this.panel3.Controls.Add(this.BtnCheckModels);
            this.panel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel3.Location = new System.Drawing.Point(3, 40);
            this.panel3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(705, 38);
            this.panel3.TabIndex = 1;
            // 
            // ModelStatusLabel
            // 
            this.ModelStatusLabel.AutoSize = true;
            this.ModelStatusLabel.Location = new System.Drawing.Point(3, 8);
            this.ModelStatusLabel.Name = "ModelStatusLabel";
            this.ModelStatusLabel.Size = new System.Drawing.Size(127, 15);
            this.ModelStatusLabel.TabIndex = 4;
            this.ModelStatusLabel.Text = "model not ready";
            // 
            // BtnRecognitionFiles
            // 
            this.BtnRecognitionFiles.BackColor = System.Drawing.SystemColors.Highlight;
            this.BtnRecognitionFiles.ForeColor = System.Drawing.SystemColors.HighlightText;
            this.BtnRecognitionFiles.Location = new System.Drawing.Point(515, 6);
            this.BtnRecognitionFiles.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BtnRecognitionFiles.Name = "BtnRecognitionFiles";
            this.BtnRecognitionFiles.Size = new System.Drawing.Size(132, 25);
            this.BtnRecognitionFiles.TabIndex = 3;
            this.BtnRecognitionFiles.Text = "Select Files";
            this.BtnRecognitionFiles.UseVisualStyleBackColor = false;
            // 
            // BtnRecognitionExample
            // 
            this.BtnRecognitionExample.BackColor = System.Drawing.SystemColors.Highlight;
            this.BtnRecognitionExample.ForeColor = System.Drawing.SystemColors.HighlightText;
            this.BtnRecognitionExample.Location = new System.Drawing.Point(421, 7);
            this.BtnRecognitionExample.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BtnRecognitionExample.Name = "BtnRecognitionExample";
            this.BtnRecognitionExample.Size = new System.Drawing.Size(90, 25);
            this.BtnRecognitionExample.TabIndex = 2;
            this.BtnRecognitionExample.Text = "Example";
            this.BtnRecognitionExample.UseVisualStyleBackColor = false;
            // 
            // BtnDeleteModels
            // 
            this.BtnDeleteModels.Location = new System.Drawing.Point(326, 6);
            this.BtnDeleteModels.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BtnDeleteModels.Name = "BtnDeleteModels";
            this.BtnDeleteModels.Size = new System.Drawing.Size(90, 25);
            this.BtnDeleteModels.TabIndex = 1;
            this.BtnDeleteModels.Text = "Delete";
            this.BtnDeleteModels.UseVisualStyleBackColor = true;
            // 
            // BtnDownLoadModels
            // 
            this.BtnDownLoadModels.Location = new System.Drawing.Point(231, 6);
            this.BtnDownLoadModels.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BtnDownLoadModels.Name = "BtnDownLoadModels";
            this.BtnDownLoadModels.Size = new System.Drawing.Size(90, 25);
            this.BtnDownLoadModels.TabIndex = 0;
            this.BtnDownLoadModels.Text = "DownLoad";
            this.BtnDownLoadModels.UseVisualStyleBackColor = true;
            // 
            // BtnCheckModels
            // 
            this.BtnCheckModels.Location = new System.Drawing.Point(137, 7);
            this.BtnCheckModels.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BtnCheckModels.Name = "BtnCheckModels";
            this.BtnCheckModels.Size = new System.Drawing.Size(90, 25);
            this.BtnCheckModels.TabIndex = 0;
            this.BtnCheckModels.Text = "Check";
            this.BtnCheckModels.UseVisualStyleBackColor = true;
            this.BtnCheckModels.Click += BtnCheckModels_Click;
            // 
            // DownloadProgressLabel
            // 
            this.DownloadProgressLabel.AutoSize = true;
            this.DownloadProgressLabel.Dock = System.Windows.Forms.DockStyle.Fill;
            this.DownloadProgressLabel.Location = new System.Drawing.Point(3, 80);
            this.DownloadProgressLabel.Name = "DownloadProgressLabel";
            this.DownloadProgressLabel.Size = new System.Drawing.Size(705, 38);
            this.DownloadProgressLabel.TabIndex = 1;
            this.DownloadProgressLabel.Text = "...";
            // 
            // OfflineAsr
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.Window;
            this.ClientSize = new System.Drawing.Size(711, 338);
            this.Controls.Add(this.tableLayoutPanel2);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Name = "OfflineAsr";
            this.Text = "SensevoiceOfflineAsr";
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel3.ResumeLayout(false);
            this.panel3.PerformLayout();
            this.ResumeLayout(false);

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