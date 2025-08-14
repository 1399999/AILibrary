namespace AILibrary;

public static class NueronUtilities
{
    public static float[][][] MatrixIndexInto(this int[][] indexer, float[][] values) // 27x2 @ 32x3
    {
        float[][][] output = new float[indexer.Length][][];

        for (int i = 0; i < indexer.Length; i++)
        {
            output[i] = new float[indexer[0].Length][];

            for (int j = 0; j < indexer[0].Length; j++)
            {
                output[i][j] = new float[values[0].Length];
            }
        }

        for (int i = 0; i < indexer.Length; i++)
        {
            if (i >= values.Length)
            {
                break;
            }

            for (int j = 0; j < indexer[i].Length; j++)
            {
                for (int k = 0; k < values[i].Length; k++)
                {
                    output[i][j][k] = values[indexer[i][j]][k];
                }
            }
        }

        return output;
    }


    public static float[][] Flatten3DTo2DArrayZToY(this float[][][] array3D) // 32x3x2 -> 32x6
    {
        if (array3D == null || array3D.Length == 0)
        {
            return Array.Empty<float[]>();
        }

        int depth = array3D.Length;
        int rowsPerDepth = array3D[0].Length;
        int cols = array3D[0][0].Length;

        int newY = rowsPerDepth * cols;

        float[][] output = new float[depth][]; // [depth][rowsPerDepth * cols]

        for (int i = 0; i < depth; i++)
        {
            output[i] = new float[newY];

            for (int j = 0; j < rowsPerDepth; j++)
            {
                for (int k = 0; k < cols; k++)
                {
                    output[i][(j * cols) + k] = array3D[i][j][k];
                }
            }
        }

        return output;
    }

    public static float[][] MatrixMultiply(this float[][] array1, float[][] array2)
    {
        if (array1 == null || array2 == null)
            throw new ArgumentNullException("Matrices cannot be null");

        int rowsA = array1.Length;
        int colsA = array1[0].Length;
        int rowsB = array2.Length;
        int colsB = array2[0].Length;

        if (colsA != rowsB)
            throw new ArgumentException("Number of columns in array1 must match number of rows in array2.");

        float[][] output = new float[rowsA][];

        for (int i = 0; i < rowsA; i++)
        {
            output[i] = new float[colsB];

            for (int j = 0; j < colsB; j++)
            {
                float sum = 0;

                for (int k = 0; k < colsA; k++)
                {
                    sum += array1[i][k] * array2[k][j];
                }

                output[i][j] = sum;
            }
        }

        return output;
    }

    public static float[][] OffsetArray(this float[][] array1, float[] array2) // Dimenions: 32x100 + 100
    {
        float[][] output = new float[array1.Length][];
        Array.Copy(array1, output, array1.Length);

        if (array1[0].Length != array2.Length)
        {
            throw new ArgumentException();
        }

        for (int i = 0; i < array1.Length; i++) 
        {
            for (int j = 0; j < array2.Length; j++)
            {
                output[i][j] += array2[j];
            }
        }

        return output;
    }

    public static float[][] GetTanh(this float[][] input)
    {
        if (input == null || input.Length == 0)
        {
            return Array.Empty<float[]>();
        }

        int rows = input.Length;
        int cols = input[0].Length;

        float[][] output = new float[rows][];

        for (int i = 0; i < rows; i++)
        {
            output[i] = new float[cols];

            for (int j = 0; j < cols; j++)
            {
                float x = input[i][j];
                // More numerically stable than direct formula for large x
                double e2x = Math.Exp(2 * x);
                output[i][j] = (float)((e2x - 1) / (e2x + 1));
            }
        }

        return output;
    }

    public static float[][] Exponentiate(this float[][] input)
    {
        float[][] output = new float[input.Length][];
        Array.Copy(input, output, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            for (int j = 0; j < input[i].Length; j++)
            {
                output[i][j] = (float) Math.Log(input[i][j]);
            }
        }

        return output;
    }

    public static float[][] DivideArray(this float[][] input, float[] divider)
    {
        float[][] output = new float[input.Length][];
        Array.Copy(input, output, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            for (int j = 0; j < input[i].Length; j++)
            {
                output[i][j] = input[i][j] / divider[j];
            }
        }

        return output;
    }

    public static float[] GetArraySum(this float[][] input)
    {
        float[] output = new float[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            for (int j = 0; j < input[i].Length; j++)
            {
                output[i] += input[i][j];
            }
        }

        return output;
    }

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

    public static long Nelement(this float[][] array) => (long)array.Length * array[0].Length;
    public static long Nelement(this float[] array) => (long)array.Length;
}
