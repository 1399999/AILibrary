namespace AILibrary.Framework;

public static class TensorUtilities
{
    public static float[] Float(this int[] array)
    {
        float[] output = new float[array.Length];

        for (int i = 0; i < array.Length; i++)
        {
            output[i] = array[i];
        }

        return output;
    }

    public static float[][] Float(this int[][] array)
    {
        float[][] output = new float[array.Length][];

        for (int i = 0; i < array.Length; i++)
        {
            output[i] = new float[array[i].Length];

            for (int j = 0; j < array[i].Length; j++)
            {
                output[i][j] = array[i][j];
            }
        }

        return output;
    }

    public static Tensor Ones(this int[] shape)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        for (int i = 0; i < expandedShape; i++)
        {
            data[i] = 1;
        }

        return new Tensor(new IntermediateArray(data, shape));
    }

    public static Tensor Zeros(this int[] shape)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        return new Tensor(new IntermediateArray(data, shape));
    }

    public static long Nelement(this Tensor tensor)
    {
        long output = 1;

        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            output *= tensor.Shape[i];
        }

        return output;
    }

    public static Tensor Tanh(this Tensor tensor) 
    {
        var negTensor = -tensor;

        var expx = tensor.Exp();
        var negexpx = negTensor.Exp();

        var a = expx - negexpx;
        var b = expx + negexpx;

        return a / b;
    }

    public static Tensor CrossEntropy(this Tensor logits, IntermediateArray labels)
    {
        var counts = logits.Exp();
        var prob = counts / counts.Sum(0, keepdims: true);

        var abcd = prob[[ArangeInt(32), labels]];
        var efgh = abcd.Log();
        //var ijkl = efgh.Mean();

        //return -ijkl;
        return -efgh;
    }

    public static IntermediateArray ArangeInt(this int ender) // 0 -> ender
    {
        float[] ints = new float[ender];

        for (int i = 0; i < ints.Length; i++)
        {
            ints[i] = (float)i;
        }

        return new IntermediateArray(ints);
    }

    public static IntermediateArray Expand(this IntermediateArray array, int times)
    {
        float[] floats = new float[times * array.InternalData.Length];

        for (int i = 0; i < times; i++)
        {
            for (int j = 0; j < array.InternalData.Length; j++)
            {
                floats[(i * array.InternalData.Length) + j] = array.InternalData[j];
            }
        }

        return new IntermediateArray(floats, new int[] { array.InternalData.Length, times });
    }

    public static (IntermediateArray, IntermediateArray) Broadcast(this IntermediateArray a, IntermediateArray b)
    {
        if (a.InternalData.Length > b.InternalData.Length)
        {
            b = b.Expand(a.InternalData.Length / b.InternalData.Length);
        }

        else
        {
            a = a.Expand(b.InternalData.Length / a.InternalData.Length);
        }

        return (a, b);
    }

    public static IntermediateArray Value(this float value, int[] shape)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        for (int i = 0; i < expandedShape; i++)
        {
            data[i] = value;
        }

        return new IntermediateArray(data, shape);
    }

    public static string ListToString(this int[] array)
    {
        string output = string.Empty;

        for (int i = 0; i < array.Length; i++)
        {
            output += array[i] + ", ";
        }

        return output.Substring(0, output.Length - 2);
    }

    public static string ListToString(this float[] array)
    {
        string output = string.Empty;

        for (int i = 0; i < array.Length; i++)
        {
            output += array[i] + ", ";
        }

        return output.Substring(0, output.Length - 2);
    }
}
