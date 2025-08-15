namespace AILibrary.NeuralNetworkFramework;

public class Tensor
{
    public float? ZeroDimValue { get; set; } = null;
    public float[]? OneDimValue { get; set; } = null;
    public float[][]? TwoDimValue { get; set; } = null;
    public float[][][]? ThreeDimValue { get; set; } = null;
    public int Dimensions { get; set; }

    public Tensor(float value)
    {
        ZeroDimValue = value;
        Dimensions = 0;
    }

    public Tensor(float[] list)
    {
        OneDimValue = list;
        Dimensions = 1;
    }

    public Tensor(float[][] list)
    {
        TwoDimValue = list;
        Dimensions = 2;
    }

    public Tensor(float[][][] list)
    {
        ThreeDimValue = list;
        Dimensions = 3;
    }

    public object GetValue()
    {
        if (ZeroDimValue != null)
        {
            return ZeroDimValue;
        }

        else if (OneDimValue != null)
        {
            return OneDimValue;
        }

        else if (TwoDimValue != null)
        {
            return TwoDimValue;
        }

        else if (ThreeDimValue != null)
        {
            return ThreeDimValue;
        }

        else
        {
            throw new ArgumentException();
        }
    }
}
