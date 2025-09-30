namespace ManySpeech.AliParaformerAsr.Examples.Entities
{
    public class AsrResultEntity
    {
        public string Text { get; set; }
        public string[] Tokens { get; set; }
        public int[][] Timestamps { get; set; }
        public string[] Languages { get; set; }
        public int Index { get; set; }
        public string ModelName { get; set; }
        public double ProcessingTimeMs { get; set; }
    }
}