// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

using Microsoft.ML.OnnxRuntime.Tensors;

namespace ManySpeech.MoonshineAsr.Model
{
    public class UncachedDecodeOutputEntity
    {
        private Tensor<float>? _reversibleEmbedding;
        private List<float[]> _cacheList=new List<float[]>();

        public Tensor<float>? ReversibleEmbedding { get => _reversibleEmbedding; set => _reversibleEmbedding = value; }
        public List<float[]> CacheList { get => _cacheList; set => _cacheList = value; }
    }
}
