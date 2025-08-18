using System.Numerics;

namespace AILibrary.Temp;

public class IntermediateArray
{
    public float? DataZeroDimArray { get; set; }
    public List<float>? DataOneDimArray { get; set; }
    public List<List<float>>? DataTwoDimArray { get; set; }
    public List<List<List<float>>>? DataThreeDimArray { get; set; }
    public int Dimensions { get; set; }
    public List<int> Shape { get; set; }

    public IntermediateArray(object data, int dimensions)
    {
        Dimensions = dimensions;
        Shape = new List<int>();

        if (dimensions == 0)
        {
            DataZeroDimArray = (float)data;
            
        }

        else if (dimensions == 1)
        {
            DataOneDimArray = (List<float>)data;
            Shape.Add(DataOneDimArray.Count);
        }

        else if (dimensions == 2)
        {
            DataTwoDimArray = (List<List<float>>)data;
            Shape.Add(DataTwoDimArray.Count);
            Shape.Add(DataTwoDimArray[0].Count);
            
        }

        else if (dimensions == 3)
        {
            DataThreeDimArray = (List<List<List<float>>>)data;
            Shape.Add(DataThreeDimArray.Count);
            Shape.Add(DataThreeDimArray[0].Count);
            Shape.Add(DataThreeDimArray[0][0].Count);
        }
    }

    public static IntermediateArray operator +(IntermediateArray self, IntermediateArray other)
    {
        if (self.Dimensions == other.Dimensions)
        {
            if (self.DataZeroDimArray != null)
            {
                self.DataZeroDimArray += other.DataZeroDimArray;
            }

            else if (self.DataOneDimArray != null && other.DataOneDimArray != null)
            {
                for (int i = 0; i < self.DataOneDimArray.Count; i++)
                {
                    self.DataOneDimArray[i] += other.DataOneDimArray[i];
                }
            }

            else if (self.DataTwoDimArray != null && other.DataTwoDimArray != null)
            {
                for (int i = 0; i < self.DataTwoDimArray.Count; i++)
                {
                    for (int j = 0; j < self.DataTwoDimArray[i].Count; j++)
                    {
                        self.DataTwoDimArray[i][j] += other.DataTwoDimArray[i][j];
                    }
                }
            }

            else if (self.DataThreeDimArray != null && other.DataThreeDimArray != null)
            {
                for (int i = 0; i < self.DataThreeDimArray.Count; i++)
                {
                    for (int j = 0; j < self.DataThreeDimArray[i].Count; j++)
                    {
                        for (int k = 0; k < self.DataThreeDimArray[i][j].Count; k++)
                        {
                            self.DataThreeDimArray[i][j][k] += other.DataThreeDimArray[i][j][k];
                        }
                    }
                }
            }
        }

        return self;
    }
}
