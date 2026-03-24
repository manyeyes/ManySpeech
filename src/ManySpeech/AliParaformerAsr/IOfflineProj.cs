using ManySpeech.AliParaformerAsr.Model;
using Microsoft.ML.OnnxRuntime;

namespace ManySpeech.AliParaformerAsr
{
    internal interface IOfflineProj
    {
        OfflineModel OfflineModel 
        { 
            get; 
            set; 
        }
        public void Infer(List<OfflineInputEntity> modelInputs, List<List<int>> tokenIdsList, List<List<int[]>> timestampsList, List<string>? languages = null, List<string>? regions = null);
        internal void Dispose();
    }
}
