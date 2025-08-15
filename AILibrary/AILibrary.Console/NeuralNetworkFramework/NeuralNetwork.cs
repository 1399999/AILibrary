namespace AILibrary.NeuralNetworkFramework;

public static class NeuralNetwork
{
    public static float CrossEntropy(this float[][] predictions, int[] labels)
    {
        if (predictions == null || labels == null)
        {
            throw new ArgumentNullException("Arguments cannot be null.");
        }

        int numSamples = predictions.Length;

        if (numSamples != labels.Length)
        {
            throw new ArgumentException("Number of samples in predictions and labels must match.");
        }

        int numClasses = predictions[0].Length;
        float epsilon = 1e-15f; // to avoid log(0)

        double totalLoss = 0.0;

        for (int i = 0; i < numSamples; i++)
        {
            int label = labels[i];

            if (label < 0 || label >= numClasses)
            {
                throw new ArgumentException($"Label {label} is out of range for sample {i}.");
            }

            float p = predictions[i][label];
            p = Math.Clamp(p, epsilon, 1.0f - epsilon); // numerical stability

            totalLoss += -Math.Log(p);
        }

        return (float)(totalLoss / numSamples);
    }
}
