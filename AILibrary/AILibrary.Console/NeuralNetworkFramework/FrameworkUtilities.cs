namespace AILibrary.NeuralNetworkFramework;

public static class FrameworkUtilities
{
    public static Tensor Zeros(int xDim)
    {
        float[] output = new float[xDim];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = 0F;
        }

        return new Tensor(output);
    }

    public static Tensor Zeros(int xDim, int yDim)
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

        return new Tensor(output);
    }

    public static Tensor Zeros(int xDim, int yDim, int zDim)
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

        return new Tensor(output);
    }

    public static Tensor Ones(int xDim)
    {
        float[] output = new float[xDim];

        for (int i = 0; i < xDim; i++)
        {
            output[i] = 0F;
        }

        return new Tensor(output);
    }

    public static Tensor Ones(int xDim, int yDim)
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

        return new Tensor(output);
    }

    public static Tensor Ones(int xDim, int yDim, int zDim)
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

        return new Tensor(output);
    }
}
