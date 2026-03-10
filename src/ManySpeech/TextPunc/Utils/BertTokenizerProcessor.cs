using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.TextPunc.Utils
{
    internal class BertTokenizerProcessor : ITextProcessor
    {
        private readonly Utils.BertTokenizerHelper _bertTokenizerHelper;

        public BertTokenizerProcessor(string tokensFilePath)
        {
            _bertTokenizerHelper = new Utils.BertTokenizerHelper(tokensFilePath);
        }

        public (string[] splitText, int[] split_text_id) ProcessText(string text)
        {
            // 原有Bert处理逻辑
            string[] tokens = _bertTokenizerHelper.Tokenize(text).ToArray();
            string[] splitText = tokens.Skip(1).Take(tokens.Length - 2).ToArray();
            int[] split_text_id = _bertTokenizerHelper.Encode(text).ToTuple().Item1;
            return (splitText, split_text_id);
        }
    }
}
