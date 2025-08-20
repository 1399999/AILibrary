namespace AILibrary.Examples;

public static class TrainedAIWordGenerator
{
    static int blockSize = 3;

    public static void GenerateWord()
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

            for (int j = 0; j < blockSize; j++)
            {
                word += SystemModel.Alphabet[0];
            }

            word += words[i];
            word += SystemModel.Alphabet[0];

            for (int j = 0; j < words[i].Length + 1; j++, l++)
            {
                blockSizeWordsTemp[l] = new int[blockSize];

                for (int k = 0; k < blockSize; k++)
                {
                    blockSizeWordsTemp[l][k] = SystemModel.AlphabetNumbers[word[j + k]];
                }
            }
        }

        Tensor blockSizeWords = new Tensor(blockSizeWordsTemp.Float());

        Tensor neuralNet = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, 2, true, int.MaxValue); // C

        var weights1 = RandomNeuron.CreateRandomNeurons(blockSize * 2, 100, true, int.MaxValue); // W1
        var biases1 = RandomNeuron.CreateRandomNeurons(100, true, int.MaxValue); // b1
        var weights2 = RandomNeuron.CreateRandomNeurons(100, SystemModel.Alphabet.Length, true, int.MaxValue); // W2
        var biases2 = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, true, int.MaxValue); // b2 

        long paramaters = neuralNet.Nelement() + weights1.Nelement() + biases1.Nelement() + weights2.Nelement() + biases2.Nelement();
        Console.WriteLine(paramaters);

        var emb = blockSizeWords.IndexInto(neuralNet);
        var kiloList = emb.Reshape([-1, 6]);
        var megaList = kiloList.Matmul(weights1);
        var gigaList = megaList + biases1;
        var tanhList = gigaList.Tanh(); // h

        var logits = tanhList.Matmul(weights2) + biases2;
        float loss = logits.CrossEntropy(allWords).Data[0];

        Console.WriteLine(loss);
    }
}
