namespace AILibrary;

public static class AIWordGenerator
{
    static int blockSize = 3;

    public static void GenerateWord()
    {
        string[] words = File.ReadAllLines("C:\\AITrainingSets\\Names.txt");

        List<int> allWordsTemp = new List<int>();

        for (int i = 0; i < words.Length; i++)
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
            Console.WriteLine(words[i]);

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
                    Console.Write(word[j + k]);
                    blockSizeWords[l][k] = SystemModel.AlphabetNumbers[word[j + k]];
                }

                Console.Write(" ---> ");
                Console.WriteLine(word[j + blockSize]);
            }
        }

        float[][] nueralNet = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, 2, false); // C

        var emb = blockSizeWords.MatrixIndexInto(nueralNet);

        var weights1 = RandomNeuron.CreateRandomNeurons(blockSize * 2, 100, false); // W1
        var biases = RandomNeuron.CreateRandomNeurons(100, false); // b1

        var kiloList = emb.Flatten3DTo2DArrayZToY();
        var megaList = kiloList.MatrixMultiply(weights1);
        var gigaList = megaList.OffsetBy(biases);
        var tanhList = gigaList.GetTanh(); // h
    }
}
