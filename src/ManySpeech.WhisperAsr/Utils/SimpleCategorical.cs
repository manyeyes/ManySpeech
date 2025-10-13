namespace ManySpeech.WhisperAsr.Utils
{
    internal class SimpleCategorical
    {
        private Dictionary<int, float> probabilities;

        public SimpleCategorical(Dictionary<int, float> probabilities)
        {
            this.probabilities = probabilities;//.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            double sumProbabilities = probabilities.Values.Sum();
            if (Math.Abs(1.0 - sumProbabilities) > 1e-6)
            {
                //throw new ArgumentException("Probabilities must sum to 1.");
            }
        }

        public int Sample()
        {
            double cumulativeProbability = 0.0;
            double randomValue = Random.NextDouble();

            foreach (var kvp in probabilities)
            {
                cumulativeProbability += kvp.Value;
                if (randomValue < cumulativeProbability)
                    return kvp.Key;
            }

            throw new InvalidOperationException("Should never happen.");
        }

        private static Random Random { get; } = new Random();

        public static int Sample(float[] logits, double temperature)
        {
            // Step 1: Apply temperature scaling.  
            float[] scaledLogits = logits.Select(l => (float)(l / temperature)).ToArray();

            // Step 2: Apply the softmax function.
            float[] probabilities = ComputeHelper.SoftmaxCompute(scaledLogits); // Calculate the probability

            // Step 3: Sample according to the probability distribution.  
            double randomValue = Random.NextDouble(); // Generate a random number between 0 and 1 (exclusive).  
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += probabilities[i];
                if (randomValue <= cumulativeProbability)
                {
                    return i; // Return the selected index.
                }
            }

            // If the sum of all probabilities is less than 1 (due to floating-point precision issues), then select the last category.  
            return logits.Length - 1;
        }
    }
}