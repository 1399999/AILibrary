namespace AILibrary;

public static class NueronUtilities
{
    public static float[][][] MatrixMultiplyIndexInto(this int[][] indexer, float[][] values) // 27x2 @ 32x3
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
}
