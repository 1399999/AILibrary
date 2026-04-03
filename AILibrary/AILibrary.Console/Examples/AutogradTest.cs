namespace AILibrary.Examples;

public static class AutogradTest
{
    public static void Test(int seed = int.MaxValue)
    {
        // Instantiate input and output:
        Tensor x = TensorUtilities.CreateRandomNeurons([8, 4, 5], false);
        IntermediateArray y = TensorUtilities.RandInt(0, 50, [8, 4]);

        Tensor w1 = new Tensor(TensorUtilities.CreateRandomNeuronsArray([8, 5, 128], false) / (float)Math.Sqrt(5));
        Tensor w2 = new Tensor(TensorUtilities.CreateRandomNeuronsArray([8, 128, 128], false) / (float)Math.Sqrt(128));
        Tensor w3 = new Tensor(TensorUtilities.CreateRandomNeuronsArray([8, 128, 50], false) / (float)Math.Sqrt(128));

        Tensor? z = null;

        for (int i = 0; i < 400; i++)
        {
            z = x.Matmul(w1);
            z = z.ReLU();
            z = z.Matmul(w2);
            z = z.ReLU();
            z = z.Matmul(w3);

            var loss = z.CrossEntropy(y);

            Console.WriteLine($"{i}: {loss.ToItem().Substring(2, loss.ToItem().Length - 4)}");

            loss.Backward();

            w1 = w1 - new Tensor(w1.Grad * 0.005F);
            w2 = w2 - new Tensor(w2.Grad * 0.005F);
            w3 = w3 - new Tensor(w3.Grad * 0.005F);

            loss.ZeroGradTree();
        }
    }
}
