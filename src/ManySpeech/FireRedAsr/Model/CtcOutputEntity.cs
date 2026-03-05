// See https://github.com/manyeyes for more information
// Copyright (c)  2026 by manyeyes
namespace ManySpeech.FireRedAsr.Model
{
    public class CtcOutputEntity
    {
        private List<List<float[]>>? _logitsList;

        public List<List<float[]>>? LogitsList { get => _logitsList; set => _logitsList = value; }
    }
}
