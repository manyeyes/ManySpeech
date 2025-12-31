using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.WinForms.Sample.Model
{
    public enum RecognizerCategory
    {
        AliParaformerAsr, // paraformer,sensevoice-small model
        FireRedAsr,    // firered model
        K2TransducerAsr, // k2transducer model
        MoonshineAsr, // moonshine model
        WenetAsr, // wenet model
        WhisperAsr, // whisper model
    }
    
}
