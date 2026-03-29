namespace AILibrary.Examples;

public static class NameGenerator
{
    static int BLOCK_SIZE = 3;
    static int STEP_SIZE = 1;

    public static void GenerateWord(int seed = int.MaxValue)
    {
        string[] words = File.ReadAllLines("C:\\AITrainingSets\\Names.txt");

        //Console.WriteLine(blockSizeWords.Print(0, 8)); // X, CORRECT
        //Console.WriteLine(allWords.Print(0, 8)); // Y, CORRECT

        int n1 = (int)(0.8D * words.Length);
        int n2 = (int)(0.9D * words.Length);

        var tempTr = BuildDataset(words[0..n1]);
        var tempDev = BuildDataset(words[n1..n2]);
        var tempTe = BuildDataset(words[n2..]);

        Tensor xtr = tempTr.Item1; 
        Tensor ytr = tempTr.Item2;
        Tensor xdev = tempDev.Item1; 
        Tensor ydev = tempDev.Item2;
        Tensor xte = tempTe.Item1; 
        Tensor yte = tempTe.Item2;

        //Console.WriteLine(xtr.Print(0, 8)); // CORRECT
        //Console.WriteLine(ytr.Print(0, 8)); // CORRECT
        //Console.WriteLine(xdev.Print(0, 8)); // CORRECT
        //Console.WriteLine(ydev.Print(0, 8)); // CORRECT
        //Console.WriteLine(xte.Print(0, 8)); // CORRECT
        //Console.WriteLine(yte.Print(0, 8)); // CORRECT

        //Tensor neuralNet = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, 10, true, seed); // C
        Tensor neuralNet = new Tensor(new IntermediateArray(SystemModel.NeuralNet)); // C

        //Tensor weights1 = RandomNeuron.CreateRandomNeurons(30, 200, true, seed); // W1
        //Tensor biases1 = RandomNeuron.CreateRandomNeurons(200, true, seed); // b1
        //Tensor weights2 = RandomNeuron.CreateRandomNeurons(200, SystemModel.Alphabet.Length, true, seed); // W2
        //Tensor biases2 = RandomNeuron.CreateRandomNeurons(SystemModel.Alphabet.Length, true, seed); // b2 
        Tensor weights1 = new Tensor(SystemModel.Weights1); // W1
        Tensor biases1 = new Tensor(SystemModel.Biases1); // b1
        Tensor weights2 = new Tensor(SystemModel.Weights2); // W2
        Tensor biases2 = new Tensor(SystemModel.Biases2); // b2

        //Console.WriteLine(weights1.Print(0, 1)); // CORRECT
        //Console.WriteLine(biases1.Print(0, 1)); // CORRECT
        //Console.WriteLine(weights2.Print(0, 1)); // CORRECT
        //Console.WriteLine(biases2.Print(0, 1)); // CORRECT

        //long paramaters = neuralNet.Nelement() + weights1.Nelement() + biases1.Nelement() + weights2.Nelement() + biases2.Nelement();
        //Console.WriteLine(paramaters); // CORRECT

        Tensor lre = TensorUtilities.Linspace(-3, 0, 1000);

        //Console.WriteLine(lre.Print(0, 8)); // CORRECT

        Tensor lrs = new Tensor(10) ^ lre;

        //Console.WriteLine(lrs.Print(0, 8)); // CORRECT

        var lri = new int[STEP_SIZE];
        var lossi = new float[STEP_SIZE];
        var stepi = new int[STEP_SIZE];

        for (int i = 0; i < 1; i++)
        {
            // minibatch construct

            //Tensor ix = new Tensor(IntermediateArray.RandInt(0, xtr.Shape[0], new int[] { 32 }));
            Tensor ix = new Tensor(SystemModel.Ix);
            //Console.WriteLine(ix); // CORRECT
        }

        //var emb = blockSizeWords.IndexInto(neuralNet.Data);
        //var kiloList = emb.Reshape([-1, 6]);
        //var megaList = kiloList.Matmul(weights1);
        //var gigaList = megaList + biases1;
        //var tanhList = gigaList.Tanh(); // h

        //var tempLogits = tanhList.Matmul(weights2);
        //var logits = tempLogits + biases2;
        //Tensor loss = logits.CrossEntropy(allWords.Data);

        //loss.Backward();

        //////weights1 = weights1 - (weights1.Grad * 0.01F);
        //////weights2 = weights2 - (weights2.Grad * 0.01F);

        ////loss.ZeroGradTree();

        //Console.WriteLine(loss.Data.InternalData[0]);
    }

    static (Tensor, Tensor) BuildDataset(string[] words)
    {
        List<int> allWordsTemp = new List<int>();

        for (int i = 0; i < words.Length; i++)
        {
            for (int j = 0; j < words[i].Length; j++)
            {
                allWordsTemp.Add(SystemModel.AlphabetNumbers[words[i][j]]);
            }

            allWordsTemp.Add(0);
        }

        Tensor allWords = new Tensor(allWordsTemp.ToArray().Float()); // Y, Dimensions: <all words>

        int[][] blockSizeWordsTemp = new int[allWords.Data.Shape[0]][];

        int l = 0;

        for (int i = 0; i < words.Length; i++)
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

        Tensor blockSizeWords = new Tensor(blockSizeWordsTemp.Float()); // X, Dimensions: <all words>x<block size>

        return (blockSizeWords, allWords);
    }
}
