namespace AILibrary;

public static class AIWordGenerator
{
    static Dictionary<char, int> alphabetNumbers = new(); // stoi
    static int blockSize = 3;

    public static void GenerateWord()
    {
        for (int i = 0; i < SystemModel.Alphabet.Length; i++)
        {
            alphabetNumbers.Add(SystemModel.Alphabet[i], i);
        }

        string[] words = File.ReadAllLines("C:\\AITrainingSets\\Names.txt");

        List<int> allWordsTemp = new List<int>();

        for (int i = 0; i < words.Length; i++)
        {
            for (int j = 0; j < words[i].Length; j++)
            {
                allWordsTemp.Add(alphabetNumbers[words[i][j]]);
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
                    blockSizeWords[l][k] = alphabetNumbers[word[j + k]];
                }

                Console.Write(" ---> ");
                Console.WriteLine(word[j + blockSize]);
            }
        }


    }

    public static double[] CreateRandomNeurons(int x, bool useSeed, int seed = 0)
    {
        NormalRandom rng = useSeed ? new NormalRandom(seed) : new NormalRandom();

        double[] output = new double[x];

        for (int i = 0; i < x; i++)
        {
            output[i] = rng.NextStandardNormal();
        }

        return output;
    }

    public static double[][] CreateRandomNeurons(int x, int y, bool useSeed, int seed = 0)
    {
        NormalRandom rng = useSeed ? new NormalRandom(seed) : new NormalRandom();

        double[][] output = new double[x][];

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < x; j++)
            {
                output[i][j] = rng.NextStandardNormal();
            }
        }

        return output;
    }
}
