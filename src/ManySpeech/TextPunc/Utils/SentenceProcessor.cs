using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ManySpeech.TextPunc.Utils
{
    internal class SentenceProcessor : ITextProcessor
    {
        private readonly SentenceHelper _sentenceHelper;

        public SentenceProcessor(string tokensFilePath)
        {
            _sentenceHelper = new Utils.SentenceHelper(tokensFilePath);
        }

        public (string[] splitText, int[] split_text_id) ProcessText(string text)
        {
            // 原有Tokens处理逻辑
            string[] splitText = SentenceHelper.CodeMixSplitWords(text);
            int[] split_text_id = SentenceHelper.Tokens2ids(_sentenceHelper.Tokens, splitText);
            return (splitText, split_text_id);
        }
    }
}
