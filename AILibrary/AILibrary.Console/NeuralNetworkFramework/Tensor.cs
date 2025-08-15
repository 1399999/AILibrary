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

    public override string ToString()
    {
        if (ZeroDimValue != null)
        {
            return ZeroDimValue.ToString();
        }

        else if (OneDimValue != null)
        {
            string output = "[";

            for (int i = 0; i < OneDimValue.Length; i++)
            {
                output += OneDimValue[i].ToString();
                output += ", ";
            }

            return output + "]";
        }

        else if (TwoDimValue != null)
        {
            string output = string.Empty;

            for (int i = 0; i < TwoDimValue.Length; i++)
            {
                output += "[";

                for (int j = 0; j < TwoDimValue.Length; j++)
                {
                    output += TwoDimValue[i][j].ToString();
                    output += ", ";
                }

                output += "],\n";
            }

            return output + "]";
        }

        else if (ThreeDimValue != null)
        {
            string output = string.Empty;

            for (int i = 0; i < ThreeDimValue.Length; i++)
            {
                for (int j = 0; j < ThreeDimValue.Length; j++)
                {
                    output += "[";

                    for (int k = 0; k < ThreeDimValue.Length; k++)
                    {
                        output += ThreeDimValue[i][j][k].ToString();
                        output += ", ";
                    }

                    output += "],\n";
                }

                output += "\n";
            }

            return output + "]";
        }

        else
        {
            throw new ArgumentException();
        }
    }
}
