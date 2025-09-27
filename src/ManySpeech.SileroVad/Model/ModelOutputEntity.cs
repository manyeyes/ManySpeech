// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes

// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Collections;

namespace ManySpeech.SileroVad.Model
{
    public class ModelOutputEntity
    {

        private float[]? _modelOut;
        private List<float[]>? _modelOutStates;

        public float[]? ModelOut { get => _modelOut; set => _modelOut = value; }
        public List<float[]>? ModelOutStates { get => _modelOutStates; set => _modelOutStates = value; }
    }
}
