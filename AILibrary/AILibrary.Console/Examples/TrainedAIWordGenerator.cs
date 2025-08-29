namespace AILibrary.Examples;

public static class TrainedAIWordGenerator
{
    static int BLOCK_SIZE = 3;

    public static void GenerateWord(int seed = int.MaxValue)
    {
        string[] words = File.ReadAllLines("C:\\AITrainingSets\\Names.txt");

        List<int> allWordsTemp = new List<int>();

        for (int i = 0; i < words.Length && i < 5; i++)
        {
            for (int j = 0; j < words[i].Length; j++)
            {
                allWordsTemp.Add(SystemModel.AlphabetNumbers[words[i][j]]);
            }

            allWordsTemp.Add(0);
        }

 
        Tensor allWords = new Tensor(allWordsTemp.ToArray().Float()); // Y, Dimensions: <all words>
        int[][] blockSizeWordsTemp = new int[allWords.Data.Shape[0]][]; // X, Dimensions: <all words>x<block size>

        // Building the dataset

        int l = 0;

        for (int i = 0; i < words.Length && i < 5; i++)
        {
            string word = string.Empty;

            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                word += SystemModel.Alphabet[0];
            }

            word += words[i];
            word += SystemModel.Alphabet[0];

            for (int j = 0; j < words[i].Length + 1; j++, l++)
            {
                blockSizeWordsTemp[l] = new int[BLOCK_SIZE];

                for (int k = 0; k < BLOCK_SIZE; k++)
                {
                    blockSizeWordsTemp[l][k] = SystemModel.AlphabetNumbers[word[j + k]];
                }
            }
        }

        Tensor blockSizeWords = new Tensor(blockSizeWordsTemp.Float());

        //Tensor neuralNet = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, 2, true, seed); // C
        Tensor neuralNet = new Tensor(new IntermediateArray(SystemModel.NeuralNet)); // C

        //var weights1 = RandomNeuron.CreateRandomNeurons(BLOCK_SIZE * 2, 100, true, seed); // W1
        //var biases1 = RandomNeuron.CreateRandomNeurons(100, true, seed); // b1
        //var weights2 = RandomNeuron.CreateRandomNeurons(100, SystemModel.Alphabet.Length, true, seed); // W2
        //var biases2 = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, true, seed); // b2 
        var weights1 = new Tensor(SystemModel.Weights1);
        var biases1 = new Tensor(SystemModel.Biases1);
        var weights2 = new Tensor(SystemModel.Weights2);
        var biases2 = new Tensor(SystemModel.Biases2);

        long paramaters = neuralNet.Nelement() + weights1.Nelement() + biases1.Nelement() + weights2.Nelement() + biases2.Nelement();
        //Console.WriteLine(paramaters);

        var emb = blockSizeWords.IndexInto(neuralNet.Data);
        var kiloList = emb.Reshape([-1, 6]);
        var megaList = kiloList.Matmul(weights1);
        var gigaList = megaList + biases1;
        var tanhList = gigaList.Tanh(); // h

        var tempLogits = tanhList.Matmul(weights2);
        var logits = tempLogits + biases2;
        Tensor loss = logits.CrossEntropy(allWords.Data);

        loss.Backward();

        //weights1 = weights1 - (weights1.Grad * 0.01F);
        //weights2 = weights2 - (weights2.Grad * 0.01F);

        loss.ZeroGradTree();

        Console.WriteLine(loss.Data.InternalData[0]);
    }
}
