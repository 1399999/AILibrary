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
}
