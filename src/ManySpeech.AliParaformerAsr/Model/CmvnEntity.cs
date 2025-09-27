// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
namespace ManySpeech.AliParaformerAsr.Model
{
    internal class CmvnEntity
    {
        private List<float> _means = new List<float>();
        private List<float> _vars = new List<float>();

        public List<float> Means { get => _means; set => _means = value; }
        public List<float> Vars { get => _vars; set => _vars = value; }
    }
}
