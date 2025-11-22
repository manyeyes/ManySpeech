// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.AudioSep.Model
{
    public class Model
    {
        private string _model_type = "unet.unet";
        private Dictionary<string,string> _model_params = new Dictionary<string, string>();

        public string model_type { get => _model_type; set => _model_type = value; }
        public Dictionary<string, string> model_params { get => _model_params; set => _model_params = value; }
    }
    public class ConfEntity
    {
        private string _train_csv = "path/to/train.csv";
        private string _validation_csv = "path/to/test.csv";
        private string _model_dir = "2stems";
        private string _mix_name = "mix";
        //private string[] _instrument_list = new string[] { "vocals", "accompaniment" };
        private Dictionary<string,string> _instrument_list = new Dictionary<string, string>();
        private int _sample_rate = 44100;
        private int _frame_length = 4096;
        private int _frame_step = 1024;
        private int _T = 512;
        private int _F = 1024;
        private int _n_channels = 2;
        private int _separation_exponent = 2;
        private string _mask_extension = "zeros";
        private double _learning_rate = 1e-4;
        private int _batch_size = 4;
        private string _training_cache = "training_cache";
        private string _validation_cache = "validation_cache";
        private int _train_max_steps = 1000000;
        private int _throttle_secs = 300;
        private int _random_seed = 0;
        private int _save_checkpoints_steps = 150;
        private int _save_summary_steps = 5;
        private Model _model=new Model();

        public string train_csv { get => _train_csv; set => _train_csv = value; }
        public string validation_csv { get => _validation_csv; set => _validation_csv = value; }
        public string model_dir { get => _model_dir; set => _model_dir = value; }
        public string mix_name { get => _mix_name; set => _mix_name = value; }
        //public string[] instrument_list { get => _instrument_list; set => _instrument_list = value; }
        public Dictionary<string, string> instrument_list { get => _instrument_list; set => _instrument_list = value; }
        public int sample_rate { get => _sample_rate; set => _sample_rate = value; }
        public int frame_length { get => _frame_length; set => _frame_length = value; }
        public int frame_step { get => _frame_step; set => _frame_step = value; }
        public int T { get => _T; set => _T = value; }
        public int F { get => _F; set => _F = value; }
        public int n_channels { get => _n_channels; set => _n_channels = value; }
        public int separation_exponent { get => _separation_exponent; set => _separation_exponent = value; }
        public string mask_extension { get => _mask_extension; set => _mask_extension = value; }
        public double learning_rate { get => _learning_rate; set => _learning_rate = value; }
        public int batch_size { get => _batch_size; set => _batch_size = value; }
        public string training_cache { get => _training_cache; set => _training_cache = value; }
        public string validation_cache { get => _validation_cache; set => _validation_cache = value; }
        public int train_max_steps { get => _train_max_steps; set => _train_max_steps = value; }
        public int throttle_secs { get => _throttle_secs; set => _throttle_secs = value; }
        public int random_seed { get => _random_seed; set => _random_seed = value; }
        public int wave_checkpoints_steps { get => _save_checkpoints_steps; set => _save_checkpoints_steps = value; }
        public int save_summary_steps { get => _save_summary_steps; set => _save_summary_steps = value; }
        public Model model { get => _model; set => _model = value; }        
    }
}
