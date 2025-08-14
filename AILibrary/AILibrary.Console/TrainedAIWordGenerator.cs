namespace AILibrary;

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

        int[] allWords = allWordsTemp.ToArray(); // Y, Dimensions: <all words>
        int[][] blockSizeWords = new int[allWords.Length][]; // X, Dimensions: <all words>x<block size>

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
                blockSizeWords[l] = new int[blockSize];

                for (int k = 0; k < blockSize; k++)
                {
                    blockSizeWords[l][k] = SystemModel.AlphabetNumbers[word[j + k]];
                }
            }
        }

        float[][] neuralNet = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, 2, true, int.MaxValue); // C

        var weights1 = RandomNeuron.CreateRandomNeurons(blockSize * 2, 100, true, int.MaxValue); // W1
        var biases1 = RandomNeuron.CreateRandomNeurons(100, true, int.MaxValue); // b1
        var weights2 = RandomNeuron.CreateRandomNeurons(100, SystemModel.Alphabet.Length, true, int.MaxValue); // W2
        var biases2 = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, true, int.MaxValue); // b2 

        long paramaters = neuralNet.Nelement() + weights1.Nelement() + biases1.Nelement() + weights2.Nelement() + biases2.Nelement();
        Console.WriteLine(paramaters);

        var emb = blockSizeWords.MatrixIndexInto(neuralNet);
        var kiloList = emb.Flatten3DTo2DArrayZToY();
        var megaList = kiloList.MatrixMultiply(weights1);
        var gigaList = megaList.OffsetArray(biases1);
        var tanhList = gigaList.GetTanh(); // h

        var logits = tanhList.MatrixMultiply(weights2).OffsetArray(biases2);
        float loss = logits.CrossEntropy(allWords);

        Console.WriteLine(loss);
    }
}
