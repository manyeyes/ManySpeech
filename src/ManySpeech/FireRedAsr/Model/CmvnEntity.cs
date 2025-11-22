// See https://github.com/manyeyes for more information
// Copyright (c)  2025 by manyeyes
namespace ManySpeech.FireRedAsr.Model
{
    internal class CmvnEntity
    {
        private List<double> _means = new List<double>();
        private List<double> _vars = new List<double>();

        public List<double> Means { get => _means; set => _means = value; }
        public List<double> Vars { get => _vars; set => _vars = value; }
    }
}
