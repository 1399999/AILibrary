namespace AILibrary.Temp;

public class IntermediateArray
{
    public float? DataZeroDimArray { get; private set; }
    public List<float>? DataOneDimArray { get; private set; }
    public List<List<float>>? DataTwoDimArray { get; private set; }
    public List<List<List<float>>>? DataThreeDimArray { get; private set; }

    public IntermediateArray(object data, int dimensions)
    {
        if (dimensions == 0)
        {
            DataZeroDimArray = (float)data;
        }

        else if (dimensions == 1)
        {
            DataOneDimArray = (List<float>)data;
        }

        else if (dimensions == 2)
        {
            DataTwoDimArray = (List<List<float>>)data;
        }

        else if (dimensions == 3)
        {
            DataThreeDimArray = (List<List<List<float>>>)data;
        }
    }
}
