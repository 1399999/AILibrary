namespace AILibrary.Temp;

public class IntermediateArray
{
    public float[] InternalData { get; }
    public int[] Shape { get; }

    public IntermediateArray(float[] data, int[] shape)
    {
        int size = shape.Aggregate(1, (a, b) => a * b);
        if (size != data.Length)
            throw new ArgumentException("Data length does not match shape.");
        InternalData = data;
        Shape = shape;
    }

    // Generic constructor: accepts any nested float arrays
    public IntermediateArray(object array)
    {
        ArgumentNullException.ThrowIfNull(array);

        List<int> shapeList = new List<int>();
        List<float> flatData = new List<float>();

        ParseArray(array, shapeList, flatData, 0);

        Shape = shapeList.ToArray();
        InternalData = flatData.ToArray();
    }

    // Recursive parser
    private static void ParseArray(object array, List<int> shape, List<float> flat, int depth)
    {
        if (array is float f)
        {
            flat.Add(f);
        }
        else if (array is Array arr)
        {
            if (shape.Count <= depth)
                shape.Add(arr.Length);
            else if (shape[depth] != arr.Length)
                throw new ArgumentException("Jagged arrays must have consistent lengths.");

            foreach (var item in arr)
                ParseArray(item, shape, flat, depth + 1);
        }
        else
        {
            throw new ArgumentException("Only float or nested float arrays are supported.");
        }
    }

    // Helper: compute flat index
    private int GetFlatIndex(int[] indices)
    {
        int flatIndex = 0;
        int stride = 1;
        for (int i = Shape.Length - 1; i >= 0; i--)
        {
            flatIndex += indices[i] * stride;
            stride *= Shape[i];
        }
        return flatIndex;
    }

    public float this[params int[] indices]
    {
        get => InternalData[GetFlatIndex(indices)];
        set => InternalData[GetFlatIndex(indices)] = value;
    }

    // Expand: turn flat Data into jagged float[][]…[]
    public object GetData()
    {
        int offset = 0;
        return ExpandRecursive(Shape, ref offset);
    }

    private object ExpandRecursive(int[] shape, ref int offset)
    {
        if (shape.Length == 1)
        {
            // Base case: return a 1D float[]
            float[] arr = new float[shape[0]];
            Array.Copy(InternalData, offset, arr, 0, shape[0]);
            offset += shape[0];
            return arr;
        }
        else
        {
            // Recursive case: build jagged array
            int dim = shape[0];
            object[] arr = new object[dim];
            int[] subShape = shape.Skip(1).ToArray();
            for (int i = 0; i < dim; i++)
                arr[i] = ExpandRecursive(subShape, ref offset);
            return arr;
        }
    }

    public IntermediateArray Sum(int dim = 0, int[]? axes = null, bool keepdims = false)
    {
        if (axes == null || axes.Length == 0)
        {
            // Sum all elements.
            float total = this.InternalData.Sum();
            return new IntermediateArray(new float[] { total }, keepdims ? new int[this.Shape.Length] : new int[0]);
        }

        axes = axes.Distinct().OrderBy(a => a).ToArray();

        int[] resultShape = this.Shape.ToArray();
        foreach (int axis in axes)
        {
            if (axis < 0 || axis >= this.Shape.Length)
                throw new ArgumentException($"Invalid axis {axis} for shape {string.Join(",", this.Shape)}");
            resultShape[axis] = 1;
        }
        if (!keepdims)
            resultShape = resultShape.Where((_, i) => !axes.Contains(i)).ToArray();

        // Allocate result
        int resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1, (a, b) => a * b);
        float[] resultData = new float[resultSize];

        // Iterate over all indices of original array
        int[] indices = new int[this.Shape.Length];
        void Recurse(int dim, int flatResultIndex, int[] resultIndices)
        {
            if (dim == this.Shape.Length)
            {
                resultData[flatResultIndex] += this[indices];
                return;
            }

            for (int i = 0; i < this.Shape[dim]; i++)
            {
                indices[dim] = i;

                int nextResultIndex = flatResultIndex;
                int[] nextResultIndices = (int[])resultIndices.Clone();

                if (!axes.Contains(dim))
                {
                    int stride = 1;
                    for (int d = resultShape.Length - 1; d > 0; d--)
                        stride *= resultShape[d];

                    int idx = 0;
                    for (int j = 0; j < nextResultIndices.Length; j++)
                        idx = idx * resultShape[j] + nextResultIndices[j];

                    nextResultIndex = idx;
                }

                Recurse(dim + 1, nextResultIndex, nextResultIndices);
            }
        }

        Recurse(dim, 0, new int[resultShape.Length]);

        return new IntermediateArray(resultData, resultShape);
    }
}

public class OldIntermediateArray
{
    public float? DataZeroDimArray { get; set; }
    public List<float>? DataOneDimArray { get; set; }
    public List<List<float>>? DataTwoDimArray { get; set; }
    public List<List<List<float>>>? DataThreeDimArray { get; set; }
    public int Dimensions { get; set; }
    public List<int> Shape { get; set; }

    public OldIntermediateArray(object data, int dimensions)
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

    public static OldIntermediateArray operator +(OldIntermediateArray self, OldIntermediateArray other)
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
