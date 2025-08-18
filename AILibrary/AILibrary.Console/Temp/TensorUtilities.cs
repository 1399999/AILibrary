using System.Runtime.InteropServices;

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

    public static IntermediateArray Sqrt(this IntermediateArray array)
    {
        if (array.DataZeroDimArray != null)
        {
            array.DataZeroDimArray = (float)Math.Sqrt((double)array.DataZeroDimArray);
        }

        else if (array.DataOneDimArray != null)
        {
            for (int i = 0; i < array.DataOneDimArray.Count; i++)
            {
                array.DataOneDimArray[i] = (float)Math.Sqrt((double)array.DataOneDimArray[i]);
            }
        }

        else if (array.DataTwoDimArray != null)
        {
            for (int i = 0; i < array.DataTwoDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataTwoDimArray[i].Count; j++)
                {
                    array.DataTwoDimArray[i][j] = (float)Math.Sqrt((double)array.DataTwoDimArray[i][j]);
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
                        array.DataThreeDimArray[i][j][k] = (float)Math.Sqrt((double)array.DataThreeDimArray[i][j][k]);
                    }
                }
            }
        }

        return array;
    }

    public static float[] FlattenIntoOneDim(this IntermediateArray array)
    {
        List<float> output = new List<float>();

        if (array.DataZeroDimArray != null)
        {
            output.Add((float)array.DataZeroDimArray);
        }

        else if (array.DataOneDimArray != null)
        {
            for (int i = 0; i < array.DataOneDimArray.Count; i++)
            {
                output.Add((float)array.DataOneDimArray[i]);
            }
        }

        else if (array.DataTwoDimArray != null)
        {
            for (int i = 0; i < array.DataTwoDimArray.Count; i++)
            {
                for (int j = 0; j < array.DataTwoDimArray[i].Count; j++)
                {
                    output.Add((float)array.DataTwoDimArray[i][j]);
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
                        output.Add((float)array.DataThreeDimArray[i][j][k]);
                    }
                }
            }
        }

        return output.ToArray();
    }

    public static IntermediateArray Sum(this IntermediateArray array, int dim = 0, int[]? axes = null, bool keepdims = false) 
    {
        NDArray sum = SumInternal(new NDArray(array.FlattenIntoOneDim(), array.Shape.ToArray()), dim: dim, axes: axes, keepDims: keepdims);
        return new IntermediateArray(sum.Expand(), sum.Shape.Length);
    }

    private class NDArray
    {
        public float[] Data { get; }
        public int[] Shape { get; }

        public NDArray(float[] data, int[] shape)
        {
            int size = shape.Aggregate(1, (a, b) => a * b);
            if (size != data.Length)
                throw new ArgumentException("Data length does not match shape.");
            Data = data;
            Shape = shape;
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
            get => Data[GetFlatIndex(indices)];
            set => Data[GetFlatIndex(indices)] = value;
        }

        // Expand: turn flat Data into jagged float[][]…[]
        public object Expand()
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
                Array.Copy(Data, offset, arr, 0, shape[0]);
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
    }

    private static NDArray SumInternal(NDArray array, int dim = 0, int[]? axes = null, bool keepDims = false)
    {
        if (axes == null || axes.Length == 0)
        {
            // Sum all elements.
            float total = array.Data.Sum();
            return new NDArray(new float[] { total }, keepDims ? new int[array.Shape.Length] : new int[0]);
        }

        axes = axes.Distinct().OrderBy(a => a).ToArray();

        int[] resultShape = array.Shape.ToArray();
        foreach (int axis in axes)
        {
            if (axis < 0 || axis >= array.Shape.Length)
                throw new ArgumentException($"Invalid axis {axis} for shape {string.Join(",", array.Shape)}");
            resultShape[axis] = 1;
        }
        if (!keepDims)
            resultShape = resultShape.Where((_, i) => !axes.Contains(i)).ToArray();

        // Allocate result
        int resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1, (a, b) => a * b);
        float[] resultData = new float[resultSize];

        // Iterate over all indices of original array
        int[] indices = new int[array.Shape.Length];
        void Recurse(int dim, int flatResultIndex, int[] resultIndices)
        {
            if (dim == array.Shape.Length)
            {
                resultData[flatResultIndex] += array[indices];
                return;
            }

            for (int i = 0; i < array.Shape[dim]; i++)
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

        return new NDArray(resultData, resultShape);
    }
}
