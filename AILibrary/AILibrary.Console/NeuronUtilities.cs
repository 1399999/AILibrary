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

        float[][] result = new float[rowsA][];

        for (int i = 0; i < rowsA; i++)
        {
            result[i] = new float[colsB];

            for (int j = 0; j < colsB; j++)
            {
                float sum = 0;

                for (int k = 0; k < colsA; k++)
                {
                    sum += array1[i][k] * array2[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    public static float[][] OffsetBy(this float[][] array1, float[] array2) // Dimenions: 32x100 + 100
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
            return Array.Empty<float[]>();

        int rows = input.Length;
        int cols = input[0].Length;

        float[][] result = new float[rows][];

        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];

            for (int j = 0; j < cols; j++)
            {
                float x = input[i][j];
                // More numerically stable than direct formula for large x
                double e2x = Math.Exp(2 * x);
                result[i][j] = (float)((e2x - 1) / (e2x + 1));
            }
        }

        return result;
    }
}
