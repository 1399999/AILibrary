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

            for (int j = 0; j < array.Length; j++)
            {
                output[i][j] = array[i][j];
            }
        }

        return output;
    }

    public static Tensor Ones(this int[] shape)
    {
        int expandedShape = 0;

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
        int expandedShape = 0;

        for (int i = 0; i < shape.Length; i++)
        {
            expandedShape *= shape[i];
        }

        float[] data = new float[expandedShape];

        return new Tensor(new IntermediateArray(data, shape));
    }

    public static long Nelement(this Tensor tensor)
    {
        long output = 0;

        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            output *= tensor.Shape[i];
        }

        return output;
    }

    public static Tensor Tanh(this Tensor tensor) => (tensor.Exp() - (-tensor).Exp()) / (tensor.Exp() + (-tensor).Exp());

    public static Tensor CrossEntropy(this Tensor logits, Tensor array)
    {
        logits = logits / logits.Sum(dim: 1, keepDims: true);

        var counts = logits.Exp();
        var prob = counts / counts.Sum(1, keepDims: true);
        return ((-prob).IndexInto(ArangeInt(32)).IndexInto(array)).Log().GetMean();
    }

    public static Tensor ArangeInt(this int ender) // 0 -> ender
    {
        float[] ints = new float[ender];

        for (int i = 0; i < ints.Length; i++)
        {
            ints[i] = i;
        }

        return new Tensor(ints);
    }
}
