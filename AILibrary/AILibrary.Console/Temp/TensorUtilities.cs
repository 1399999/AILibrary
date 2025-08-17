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

    public static IntermediateArray Ones(int x)
    {
        List<float> output = new List<float>();

        for (int i = 0; i < x; i++)
        {
            output[i] = 1F;
        }

        return new IntermediateArray(output, 1);
    }

    public static IntermediateArray Ones(int x, int y)
    {
        List<List<float>> output = new List<List<float>>();

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                output[i][j] = 1F;
            }
        }

        return new IntermediateArray(output, 2);
    }

    public static IntermediateArray Ones(int x, int y, int z)
    {
        List<List<float>> output = new List<List<float>>();

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                for (int k = 0; k < z; k++)
                {
                    output[i][j] = 1F;
                }
            }
        }

        return new IntermediateArray(output, 3);
    }

    public static IntermediateArray Ones(this IntermediateArray array)
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

    public static IntermediateArray Ones(this List<int> array)
    {
        if (array.Count == 1)
        {
            return Zeros(array[0]);
        }

        else if (array.Count == 2)
        {
            return Zeros(array[0], array[1]);
        }

        else if (array.Count == 3)
        {
            return Zeros(array[0], array[1], array[2]);
        }

        else
        {
            throw new Exception();
        }
    }

    public static IntermediateArray EulExp(this IntermediateArray array)
    {
        if (array.DataZeroDimArray != null)
        {
            array.DataZeroDimArray = (float)Math.Exp((double)array.DataZeroDimArray);
        }

        else if (array.DataOneDimArray != null)
        {
            for (int i = 0; i < array.DataOneDimArray.Count; i++)
            {
                array.DataOneDimArray[i] = (float)Math.Exp((double)array.DataOneDimArray[i]);
            }
        }

        else if (array.DataTwoDimArray != null)
        {
            for (int i = 0; i < array.DataTwoDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataTwoDimArray[i].Count; j++)
                {
                    array.DataTwoDimArray[i][j] = (float)Math.Exp((double)array.DataTwoDimArray[i][j]);
                }
            }
        }

        else if (array.DataThreeDimArray != null)
        {
            for (int i = 0; i < array.DataThreeDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataThreeDimArray[i].Count; j++)
                {
                    for (int k = 0; k < array.DataThreeDimArray[i][j].Count; k++)
                    {
                        array.DataThreeDimArray[i][j][k] = (float)Math.Exp((double)array.DataThreeDimArray[i][j][k]);
                    }
                }
            }
        }

        return array;
    }

    public static IntermediateArray Log(this IntermediateArray array)
    {
        if (array.DataZeroDimArray != null)
        {
            array.DataZeroDimArray = (float)Math.Log((double)array.DataZeroDimArray);
        }

        else if (array.DataOneDimArray != null)
        {
            for (int i = 0; i < array.DataOneDimArray.Count; i++)
            {
                array.DataOneDimArray[i] = (float)Math.Log((double)array.DataOneDimArray[i]);
            }
        }

        else if (array.DataTwoDimArray != null)
        {
            for (int i = 0; i < array.DataTwoDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataTwoDimArray[i].Count; j++)
                {
                    array.DataTwoDimArray[i][j] = (float)Math.Log((double)array.DataTwoDimArray[i][j]);
                }
            }
        }

        else if (array.DataThreeDimArray != null)
        {
            for (int i = 0; i < array.DataThreeDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataThreeDimArray[i].Count; j++)
                {
                    for (int k = 0; k < array.DataThreeDimArray[i][j].Count; k++)
                    {
                        array.DataThreeDimArray[i][j][k] = (float)Math.Log((double)array.DataThreeDimArray[i][j][k]);
                    }
                }
            }
        }

        return array;
    }
}
