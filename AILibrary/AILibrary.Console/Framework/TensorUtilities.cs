using System.Xml;

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

    public static float[] OnesFloat(this int[] shape)
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

        return data;
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

    public static IntermediateArray ZerosArray(this int[] shape)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        return new IntermediateArray(data, shape);
    }

    public static float[] ZerosFloat(this int[] shape)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        return data;
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
        var negTensor = -tensor; // SUCCESS

        var expx = tensor.Exp(); // SUCCESS
        var negexpx = negTensor.Exp(); // SUCCESS

        var a = expx - negexpx;  // SUCCESS MUL AND ADD
        var b = expx + negexpx;  // SUCCESS

        return a / b;  // SUCCESS
    }

    public static Tensor CrossEntropy(this Tensor logits, IntermediateArray labels)
    {
        int tempmul = 1;

        for (int i = 0; i < logits.Shape.Length - 1; i++)
        {
            tempmul *= logits.Shape[i];
        }

        logits = logits.Reshape([tempmul, logits.Shape[^1]]);

        var counts = logits.Exp();
        var prob = counts / counts.Sum(1, keepdims: true);

        var logLosses = prob[[ArangeInt(tempmul), labels]].Log();

        return -logLosses.Sum() / tempmul;
    }

    public static Tensor CrossEntropy(this Tensor logits, Tensor labels)
    {
        int tempmul = 1;

        for (int i = 0; i < logits.Shape.Length - 1; i++)
        {
            tempmul *= logits.Shape[i];
        }

        logits = logits.Reshape([tempmul, logits.Shape[^1]]);

        var counts = logits.Exp();
        var prob = counts / counts.Sum(1, keepdims: true);

        var abcd = prob[[ArangeInt(tempmul), labels.Data]];
        var efgh = abcd.Log();
        var ijkl = efgh.Mean();

        return -ijkl;
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

    public static IntermediateArray ExpandOther(this IntermediateArray array, int times)
    {
        float[] floats = new float[times * array.InternalData.Length];

        for (int i = 0; i < array.InternalData.Length; i++)
        {
            for (int j = 0; j < times; j++)
            {
                floats[(i * times) + j] = array.InternalData[i];
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

    public static (IntermediateArray, IntermediateArray) BroadcastOther(this IntermediateArray a, IntermediateArray b)
    {
        if (a.InternalData.Length > b.InternalData.Length)
        {
            b = b.ExpandOther(a.InternalData.Length / b.InternalData.Length);
        }

        else
        {
            a = a.ExpandOther(b.InternalData.Length / a.InternalData.Length);
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

    public static float[] ValueFloat(this float value, int[] shape)
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

        return data;
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

    public static int[] RemoveItem(this int[] array, int item)
    {
        int[] output = new int[array.Length - 1];

        for (int i = 0, j = 0; i < array.Length; i++)
        {
            if (array[i] != item)
            {
                output[j] = array[i];
                j++;
            }
        }

        return output;
    }

    public static int[] RemoveIndex(this int[] array, int index)
    {
        int[] output = new int[array.Length - 1];

        for (int i = 0, j = 0; i < array.Length; i++)
        {
            if (i != index)
            {
                output[j] = array[i];
                j++;
            }
        }

        return output;
    }

    public static int[] InsertItem(this int[] array, int index, int item)
    {
        int[] output = new int[array.Length];

        for (int i = 0; i < array.Length; i++)
        {
            output[i] = array[i];

            if (i == index)
            {
                output[i] = item;
            }
        }

        return output;
    }

    public static int[] Replicate(this int integer, int times)
    {
        int[] output = new int[times];

        for (int i = 0; i < times; i++)
        {
            output[i] = integer;
        }

        return output;
    }

    public static Tensor Linspace(int num1, int num2, int totalElements)
    {
        float[] output = new float[totalElements + 1];

        int diff = num2 - num1;

        float j = 0;

        for (int i = 0; i < totalElements; i++, j += (float)diff / (float)totalElements)
        {
            output[i] = num1 + j;
        }

        output[^1] = num2;

        return new Tensor(output);
    }

    public static Tensor CreateRandomNeurons(int[] shape, bool useSeed, int seed = 0)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] output = new float[expandedShape];

        for (int i = 0; i < expandedShape; i++)
        {
            output[i] = NextStandardNormal();
        }

        return new Tensor(new IntermediateArray(output, shape));
    }

    public static IntermediateArray CreateRandomNeuronsArray(int[] shape, bool useSeed, int seed = 0)
    {
        int expandedShape = 1;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] output = new float[expandedShape];

        for (int i = 0; i < expandedShape; i++)
        {
            output[i] = NextStandardNormal();
        }

        return new IntermediateArray(output, shape);
    }

    static float NextStandardNormal()
    {
        Random _rand = new Random();

        float u, v, s;
        do
        {
            u = (float)_rand.NextDouble() * 2.0F - 1.0F; // uniform(-1, 1)
            v = (float)_rand.NextDouble() * 2.0F - 1.0F;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0);

        float multiplier = (float)Math.Sqrt(-2.0 * Math.Log(s) / s);

        return u * multiplier; // standard normal (mean=0, std=1)
    }

    public static IntermediateArray RandInt(int low, int high, params int[] shape)
    {
        int totalElements = shape.Aggregate(1, (a, b) => a * b);
        float[] data = new float[totalElements];
        Random rng = new Random();

        for (int i = 0; i < totalElements; i++)
        {
            data[i] = rng.Next(low, high);
        }

        return new IntermediateArray(data: data, shape: shape);
    }

    public static Tensor ReLU(this Tensor input)
    {
        var output = new float[input.Data.InternalData.Length];

        for (int i = 0; i < input.Data.InternalData.Length; i++)
        {
            if (input.Data.InternalData[i] > 0)
            {
                output[i] = input.Data.InternalData[i];
            }

            else
            {
                output[i] = 0;
            }
        }

        return new Tensor(new IntermediateArray(output, input.Shape));
    }
}
