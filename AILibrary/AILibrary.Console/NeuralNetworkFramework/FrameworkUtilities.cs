namespace AILibrary.NeuralNetworkFramework;

public static class FrameworkUtilities
{
    public static float[] Zeros(int xDim)
    {
        float[] output = new float[xDim];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = 0F;
        }

        return output;
    }

    public static float[][] Zeros(int xDim, int yDim)
    {
        float[][] output = new float[xDim][];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = new float[yDim];

            for (int j = 0; j < yDim; j++)
            {
                output[i][j] = 0F;
            }
        }

        return output;
    }

    public static float[][][] Zeros(int xDim, int yDim, int zDim)
    {
        float[][][] output = new float[xDim][][];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = new float[yDim][];

            for (int j = 0; j < yDim; j++)
            {
                output[i][j] = new float[zDim];

                for (int k = 0; k < yDim; k++)
                {
                    output[i][j][k] = 0F;
                }
            }
        }

        return output;
    }

    public static float[] Ones(int xDim)
    {
        float[] output = new float[xDim];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = 0F;
        }

        return output;
    }

    public static float[][] Ones(int xDim, int yDim)
    {
        float[][] output = new float[xDim][];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = new float[yDim];

            for (int j = 0; j < yDim; j++)
            {
                output[i][j] = 0F;
            }
        }

        return output;
    }

    public static float[][][] Ones(int xDim, int yDim, int zDim)
    {
        float[][][] output = new float[xDim][][];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = new float[yDim][];

            for (int j = 0; j < yDim; j++)
            {
                output[i][j] = new float[zDim];

                for (int k = 0; k < yDim; k++)
                {
                    output[i][j][k] = 0F;
                }
            }
        }

        return output;
    }
}
