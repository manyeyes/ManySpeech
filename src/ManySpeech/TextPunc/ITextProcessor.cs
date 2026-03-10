namespace ManySpeech.TextPunc
{
    public interface ITextProcessor
    {
        /// <summary>
        /// 仅传入文本，返回分词+ID结果
        /// </summary>
        /// <param name="text">待处理文本</param>
        /// <returns>(splitText, split_text_id) 元组</returns>
        (string[] splitText, int[] split_text_id) ProcessText(string text);
    }
}