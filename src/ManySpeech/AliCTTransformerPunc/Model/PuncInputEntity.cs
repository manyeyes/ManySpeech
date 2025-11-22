using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.AliCTTransformerPunc.Model
{
    public class PuncInputEntity
    {
        private int[] _miniSentenceId;
        private int _textLengths;

        public int[] MiniSentenceId { get => _miniSentenceId; set => _miniSentenceId = value; }
        public int TextLengths { get => _textLengths; set => _textLengths = value; }
    }
}
