namespace ManySpeech.WinForms.Sample
{
    partial class FormMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
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
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            leftPanel = new Panel();
            treeMenu = new TreeView();
            treeView1 = new TreeView();
            rightPanel = new Panel();
            leftPanel.SuspendLayout();
            SuspendLayout();
            // 
            // leftPanel
            // 
            leftPanel.Controls.Add(treeMenu);
            leftPanel.Controls.Add(treeView1);
            leftPanel.Dock = DockStyle.Left;
            leftPanel.Location = new Point(0, 0);
            leftPanel.Name = "leftPanel";
            leftPanel.Size = new Size(250, 450);
            leftPanel.TabIndex = 0;
            // 
            // treeMenu
            // 
            treeMenu.Location = new Point(0, 0);
            treeMenu.Name = "treeMenu";
            treeMenu.Size = new Size(250, 446);
            treeMenu.TabIndex = 1;
            treeMenu.NodeMouseClick += treeMenu_NodeMouseClick;
            // 
            // treeView1
            // 
            treeView1.Location = new Point(3, 0);
            treeView1.Name = "treeView1";
            treeView1.Size = new Size(247, 446);
            treeView1.TabIndex = 0;
            // 
            // rightPanel
            // 
            rightPanel.Dock = DockStyle.Fill;
            rightPanel.Location = new Point(250, 0);
            rightPanel.Name = "rightPanel";
            rightPanel.Size = new Size(550, 450);
            rightPanel.TabIndex = 1;
            rightPanel.Paint += rightPanel_Paint;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(9F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(rightPanel);
            Controls.Add(leftPanel);
            Name = "Form1";
            Text = "ManySpeech.WinForms.Sample";
            Load += Form1_Load;
            leftPanel.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private Panel leftPanel;
        private TreeView treeView1;
        private Panel rightPanel;
        private TreeView treeMenu;
    }
}
