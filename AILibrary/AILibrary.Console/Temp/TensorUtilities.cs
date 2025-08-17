namespace AILibrary.Temp;

public static class TensorUtilities
{
    public static IntermediateArray Zeros(int x)
    {
        List<float> output = new List<float>();

        for (int i = 0; i < x; i++)
        {
            output[i] = 0F;
        }

        return new IntermediateArray(output, 1);
    }

    public static IntermediateArray Zeros(int x, int y)
    {
        List<List<float>> output = new List<List<float>>();

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                output[i][j] = 0F;
            }
        }

        return new IntermediateArray(output, 2);
    }

    public static IntermediateArray Zeros(int x, int y, int z)
    {
        List<List<float>> output = new List<List<float>>();

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < z; k++)
                {
                    output[i][j] = 0F;
                }
            }
        }

        return new IntermediateArray(output, 3);
    }

    public static IntermediateArray Zeros(this IntermediateArray array)
    {
        if (array.DataZeroDimArray != null)
        {
            return new IntermediateArray(0, 0);
        }

        else if (array.DataOneDimArray != null)
        {
            return Zeros(array.DataOneDimArray.Count);
        }

        else if (array.DataTwoDimArray != null)
        {
            return Zeros(array.DataTwoDimArray.Count, array.DataTwoDimArray[0].Count);
        }

        else if (array.DataThreeDimArray != null)
        {
            return Zeros(array.DataThreeDimArray.Count, array.DataThreeDimArray[0].Count, array.DataThreeDimArray[0][0].Count);
        }

        else
        {
            throw new Exception();
        }
    }
}
