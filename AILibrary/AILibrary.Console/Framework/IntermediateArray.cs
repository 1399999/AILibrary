using System;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AILibrary.Framework;

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
            float total = InternalData.Sum();
            return new IntermediateArray(new float[] { total }, keepdims ? new int[Shape.Length] : new int[0]);
        }

        axes = axes.Distinct().OrderBy(a => a).ToArray();

        int[] resultShape = Shape.ToArray();
        foreach (int axis in axes)
        {
            if (axis < 0 || axis >= Shape.Length)
                throw new ArgumentException($"Invalid axis {axis} for shape {string.Join(",", Shape)}");
            resultShape[axis] = 1;
        }
        if (!keepdims)
            resultShape = resultShape.Where((_, i) => !axes.Contains(i)).ToArray();

        // Allocate result
        int resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1, (a, b) => a * b);
        float[] resultData = new float[resultSize];

        // Iterate over all indices of original array
        int[] indices = new int[Shape.Length];
        void Recurse(int dim, int flatResultIndex, int[] resultIndices)
        {
            if (dim == Shape.Length)
            {
                resultData[flatResultIndex] += this[indices];
                return;
            }

            for (int i = 0; i < Shape[dim]; i++)
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

    public IntermediateArray ZerosLike()
    {
        int size = Shape.Aggregate(1, (a, b) => a * b);
        float[] zeros = new float[size]; // Automatically filled with 0
        return new IntermediateArray(zeros, Shape.ToArray()); // Copy shape
    }

    public IntermediateArray OnesLike()
    {
        int size = Shape.Aggregate(1, (a, b) => a * b);
        float[] zeros = new float[size]; // Automatically filled with 0.

        for (int i = 0; i < zeros.Length; i++)
        {
            zeros[i] = 1;
        }

        return new IntermediateArray(zeros, Shape.ToArray()); // Copy shape
    }

    /// <summary>
    /// NumPy-like elementwise addition.
    /// </summary>
    public static IntermediateArray operator +(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise addition (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = a.InternalData[i] + b.InternalData[i];

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    // NumPy-like add() function
    public static IntermediateArray Add(IntermediateArray a, IntermediateArray b) => a + b;

    /// <summary>
    /// NumPy-like elementwise subtraction.
    /// </summary>
    public static IntermediateArray operator -(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise subtraction (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = a.InternalData[i] - b.InternalData[i];

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    public static IntermediateArray Subtract(IntermediateArray a, IntermediateArray b) => a - b;

    // -------- numpy.prod() implementation --------
    public IntermediateArray Prod(int[]? axes = null, bool keepDims = false)
    {
        if (axes == null || axes.Length == 0)
        {
            // Multiply everything
            float product = 1.0f;
            foreach (var v in InternalData)
                product *= v;

            return new IntermediateArray(new float[] { product },
                               keepDims ? new int[Shape.Length] : new int[0]);
        }

        axes = axes.Distinct().OrderBy(a => a).ToArray();

        // Validate axes
        foreach (int axis in axes)
        {
            if (axis < 0 || axis >= Shape.Length)
                throw new ArgumentException($"Invalid axis {axis} for shape [{string.Join(",", Shape)}]");
        }

        // Build result shape
        int[] resultShape = Shape.ToArray();
        foreach (var axis in axes)
            resultShape[axis] = 1;

        if (!keepDims)
            resultShape = resultShape.Where((_, i) => !axes.Contains(i)).ToArray();

        // Allocate result
        int resultSize = resultShape.Length == 0 ? 1 : resultShape.Aggregate(1, (a, b) => a * b);
        float[] resultData = new float[resultSize];

        // Initialize with 1.0 so multiplication works
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = 1.0f;

        // Iterate over all elements in original array
        int[] indices = new int[Shape.Length];

        void Recurse(int dim, int resultIndex, int[] resultIndices)
        {
            if (dim == Shape.Length)
            {
                resultData[resultIndex] *= this[indices];
                return;
            }

            for (int i = 0; i < Shape[dim]; i++)
            {
                indices[dim] = i;
                int nextResultIndex = resultIndex;
                int[] nextResultIndices = (int[])resultIndices.Clone();

                if (!axes.Contains(dim))
                {
                    // compute flattened index for resultIndices
                    int idx = 0;
                    for (int j = 0; j < nextResultIndices.Length; j++)
                        idx = idx * resultShape[j] + nextResultIndices[j];
                    nextResultIndex = idx;
                }

                Recurse(dim + 1, nextResultIndex, nextResultIndices);
            }
        }

        Recurse(0, 0, new int[resultShape.Length]);

        return new IntermediateArray(resultData, resultShape);
    }

    /// <summary>
    /// Returns a new NDArray with every element negated.
    /// </summary>
    public IntermediateArray Negate()
    {
        float[] resultData = new float[InternalData.Length];
        for (int i = 0; i < InternalData.Length; i++)
            resultData[i] = -InternalData[i];

        return new IntermediateArray(resultData, Shape.ToArray());
    }

    /// <summary>
    /// Unary minus operator for NDArray (elementwise negation).
    /// </summary>
    public static IntermediateArray operator -(IntermediateArray a)
    {
        return a.Negate();
    }

    /// <summary>
    /// NumPy-like elementwise multiplication.
    /// </summary>
    public static IntermediateArray Multiply(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise multiplication (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = a.InternalData[i] * b.InternalData[i];

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// Overload * operator for elementwise multiplication.
    /// </summary>
    public static IntermediateArray operator *(IntermediateArray a, IntermediateArray b) => Multiply(a, b);

    /// <summary>
    /// NumPy-like elementwise division.
    /// </summary>
    public static IntermediateArray Divide(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise division (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
        {
            if (b.InternalData[i] == 0)
                throw new DivideByZeroException("Division by zero encountered in NDArray.");
            resultData[i] = a.InternalData[i] / b.InternalData[i];
        }

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// Overload / operator for elementwise division.
    /// </summary>
    public static IntermediateArray operator /(IntermediateArray a, IntermediateArray b) => Divide(a, b);

    /// <summary>
    /// NumPy-like elementwise power.
    /// </summary>
    public static IntermediateArray Power(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise power (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = (float)Math.Pow(a.InternalData[i], b.InternalData[i]);

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// NumPy-like elementwise power with a scalar exponent.
    /// </summary>
    public static IntermediateArray Power(IntermediateArray a, float exponent)
    {
        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = (float)Math.Pow(a.InternalData[i], exponent);

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// Overload ^ operator for elementwise power.
    /// </summary>
    public static IntermediateArray operator ^(IntermediateArray a, IntermediateArray b) => Power(a, b);

    public static IntermediateArray operator ^(IntermediateArray a, float exponent) => Power(a, exponent);

    /// <summary>
    /// NumPy-like matrix multiplication (matmul).
    /// Only supports 2D arrays for now: (m x n) @ (n x p) = (m x p).
    /// </summary>
    public IntermediateArray Matmul(IntermediateArray b)
    {
        if (Shape.Length != 2 || b.Shape.Length != 2)
            throw new ArgumentException("Matmul currently only supports 2D arrays.");

        int m = Shape[0]; // rows of A
        int n = Shape[1]; // cols of A = rows of B
        int p = b.Shape[1]; // cols of B

        if (n != b.Shape[0])
            throw new ArgumentException("Shapes are not aligned for matrix multiplication.");

        float[] resultData = new float[m * p];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                float sum = 0f;
                for (int k = 0; k < n; k++)
                {
                    sum += InternalData[i * n + k] * b.InternalData[k * p + j];
                }
                resultData[i * p + j] = sum;
            }
        }

        return new IntermediateArray(resultData, new int[] { m, p });
    }

    /// <summary>
    /// NumPy-like SwapAxes function. 
    /// Returns a new NDArray with the given axes swapped.
    /// </summary>
    public IntermediateArray SwapAxes(int axis1, int axis2)
    {
        if (axis1 < 0 || axis1 >= Shape.Length || axis2 < 0 || axis2 >= Shape.Length)
            throw new ArgumentException("Axis index out of range.");

        if (axis1 == axis2)
            return new IntermediateArray((float[])InternalData.Clone(), (int[])Shape.Clone());

        // New shape with swapped axes
        int[] newShape = (int[])Shape.Clone();
        int temp = newShape[axis1];
        newShape[axis1] = newShape[axis2];
        newShape[axis2] = temp;

        float[] resultData = new float[InternalData.Length];

        // Recursive index walker
        void Recurse(int[] oldIndices, int depth, int[] strides, int[] newStrides)
        {
            if (depth == Shape.Length)
            {
                // Swap indices
                int[] newIndices = (int[])oldIndices.Clone();
                int tmp = newIndices[axis1];
                newIndices[axis1] = newIndices[axis2];
                newIndices[axis2] = tmp;

                int oldFlat = 0, newFlat = 0;
                for (int i = 0; i < Shape.Length; i++)
                {
                    oldFlat += oldIndices[i] * strides[i];
                    newFlat += newIndices[i] * newStrides[i];
                }
                resultData[newFlat] = InternalData[oldFlat];
                return;
            }

            for (int i = 0; i < Shape[depth]; i++)
            {
                oldIndices[depth] = i;
                Recurse(oldIndices, depth + 1, strides, newStrides);
            }
        }

        // Compute strides for old and new shapes
        int[] strides = ComputeStrides(Shape);
        int[] newStrides = ComputeStrides(newShape);

        Recurse(new int[Shape.Length], 0, strides, newStrides);

        return new IntermediateArray(resultData, newShape);
    }

    /// <summary>
    /// Helper: compute strides for row-major order
    /// </summary>
    private static int[] ComputeStrides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// NumPy-like elementwise exponential function.
    /// Returns a new NDArray where each element is e^x.
    /// </summary>
    public IntermediateArray Exp()
    {
        float[] resultData = new float[InternalData.Length];
        for (int i = 0; i < InternalData.Length; i++)
        {
            resultData[i] = (float)Math.Exp(InternalData[i]);
        }
        return new IntermediateArray(resultData, (int[])Shape.Clone());
    }

    /// <summary>
    /// NumPy-like elementwise natural logarithm (ln).
    /// Returns a new NDArray where each element is log_e(x).
    /// </summary>
    public IntermediateArray Log()
    {
        float[] resultData = new float[InternalData.Length];
        for (int i = 0; i < InternalData.Length; i++)
        {
            if (InternalData[i] <= 0)
                throw new ArgumentException("Log undefined for zero or negative values in IntermediateArray.");

            resultData[i] = MathF.Log(InternalData[i]);
        }
        return new IntermediateArray(resultData, (int[])Shape.Clone());
    }

    /// <summary>
    /// NumPy-like elementwise square root.
    /// Returns a new NDArray where each element is sqrt(x).
    /// </summary>
    public IntermediateArray Sqrt()
    {
        float[] resultData = new float[InternalData.Length];
        for (int i = 0; i < InternalData.Length; i++)
        {
            if (InternalData[i] < 0)
                throw new ArgumentException("Sqrt undefined for negative values in IntermediateArray.");

            resultData[i] = MathF.Sqrt(InternalData[i]);
        }
        return new IntermediateArray(resultData, (int[])Shape.Clone());
    }

    /// <summary>
    /// NumPy-like mean function.
    /// If axis is null, returns the mean of all elements (scalar IntermediateArray unless keepdims=true).
    /// If axis is specified, computes the mean along that axis.
    /// </summary>
    public IntermediateArray Mean(int? axis = null, bool keepdims = false)
    {
        if (axis == null)
        {
            // Global mean → single scalar
            float sum = 0f;
            for (int i = 0; i < InternalData.Length; i++)
                sum += InternalData[i];
            float mean = sum / InternalData.Length;

            int[] shape = keepdims ? Enumerable.Repeat(1, Shape.Length).ToArray() : new int[] { };
            return new IntermediateArray(new float[] { mean }, shape);
        }
        else
        {
            int ax = axis.Value;
            if (ax < 0 || ax >= Shape.Length)
                throw new ArgumentException("Axis out of range.");

            int axisSize = Shape[ax];

            // Compute strides
            int[] strides = ComputeStrides(Shape);

            // New shape
            int[] newShape;
            if (keepdims)
            {
                newShape = Shape.ToArray();
                newShape[ax] = 1;
            }
            else
            {
                newShape = Shape.Where((_, i) => i != ax).ToArray();
            }

            float[] resultData = new float[InternalData.Length / axisSize];

            // Recursive iterator
            void Recurse(int[] indices, int depth, int resultIndex)
            {
                if (depth == Shape.Length)
                    return;

                if (depth == ax)
                {
                    float sum = 0f;
                    for (int j = 0; j < axisSize; j++)
                    {
                        indices[depth] = j;
                        int flatIndex = 0;
                        for (int k = 0; k < Shape.Length; k++)
                            flatIndex += indices[k] * strides[k];
                        sum += InternalData[flatIndex];
                    }
                    resultData[resultIndex] = sum / axisSize;
                    return;
                }

                for (int i = 0; i < Shape[depth]; i++)
                {
                    indices[depth] = i;
                    int newFlat = 0, count = 0;

                    for (int k = 0; k < Shape.Length; k++)
                    {
                        if (!keepdims && k == ax) continue;
                        newFlat += indices[k] * ComputeStrides(newShape)[count++];
                    }

                    Recurse(indices, depth + 1, newFlat);
                }
            }

            Recurse(new int[Shape.Length], 0, 0);

            return new IntermediateArray(resultData, newShape);
        }
    }

    /// <summary>
    /// NumPy-like elementwise division where a scalar (float) 
    /// is divided by every element in the IntermediateArray.
    /// Equivalent to np.divide(scalar, array).
    /// </summary>
    public static IntermediateArray operator *(float scalar, IntermediateArray array)
    {
        float[] resultData = new float[array.InternalData.Length];
        for (int i = 0; i < array.InternalData.Length; i++)
        {
            if (array.InternalData[i] == 0f)
                throw new DivideByZeroException("Division by zero in IntermediateArray.");

            resultData[i] = scalar * array.InternalData[i];
        }
        return new IntermediateArray(resultData, (int[])array.Shape.Clone());
    }

    /// <summary>
    /// NumPy-like elementwise division where a scalar (float) 
    /// is divided by every element in the IntermediateArray.
    /// Equivalent to np.divide(scalar, array).
    /// </summary>
    public static IntermediateArray operator *(IntermediateArray array, float scalar)
    {
        float[] resultData = new float[array.InternalData.Length];
        for (int i = 0; i < array.InternalData.Length; i++)
        {
            if (array.InternalData[i] == 0f)
                throw new DivideByZeroException("Division by zero in IntermediateArray.");

            resultData[i] = scalar * array.InternalData[i];
        }
        return new IntermediateArray(resultData, (int[])array.Shape.Clone());
    }

    /// <summary>
    /// NumPy-like Max function.
    /// If axis is null, returns the maximum of all elements.
    /// If axis is specified, computes the max along that axis.
    /// </summary>
    public IntermediateArray Max(int? axis = null, bool keepdims = false)
    {
        if (axis == null)
        {
            float maxVal = float.NegativeInfinity;
            for (int i = 0; i < InternalData.Length; i++)
                if (InternalData[i] > maxVal)
                    maxVal = InternalData[i];

            int[] shape = keepdims ? Enumerable.Repeat(1, Shape.Length).ToArray() : new int[] { };
            return new IntermediateArray(new float[] { maxVal }, shape);
        }
        else
        {
            int ax = axis.Value;
            if (ax < 0 || ax >= Shape.Length)
                throw new ArgumentException("Axis out of range.");

            int axisSize = Shape[ax];

            // new shape
            int[] newShape;
            if (keepdims)
            {
                newShape = Shape.ToArray();
                newShape[ax] = 1;
            }
            else
            {
                newShape = Shape.Where((_, i) => i != ax).ToArray();
            }

            float[] resultData = new float[InternalData.Length / axisSize];
            int[] strides = ComputeStrides(Shape);
            int[] newStrides = ComputeStrides(newShape);

            void Recurse(int[] indices, int depth, int resultIndex)
            {
                if (depth == Shape.Length)
                    return;

                if (depth == ax)
                {
                    float maxVal = float.NegativeInfinity;
                    for (int j = 0; j < axisSize; j++)
                    {
                        indices[depth] = j;
                        int flatIndex = 0;
                        for (int k = 0; k < Shape.Length; k++)
                            flatIndex += indices[k] * strides[k];
                        if (InternalData[flatIndex] > maxVal)
                            maxVal = InternalData[flatIndex];
                    }
                    resultData[resultIndex] = maxVal;
                    return;
                }

                for (int i = 0; i < Shape[depth]; i++)
                {
                    indices[depth] = i;

                    int newFlat = 0, count = 0;
                    for (int k = 0; k < Shape.Length; k++)
                    {
                        if (!keepdims && k == ax) continue;
                        newFlat += indices[k] * newStrides[count++];
                    }

                    Recurse(indices, depth + 1, newFlat);
                }
            }

            Recurse(new int[Shape.Length], 0, 0);

            return new IntermediateArray(resultData, newShape);
        }

    }

    /// <summary>
    /// NumPy-like expand_dims function.
    /// Inserts a new axis of size 1 at the specified axis position.
    /// </summary>
    public IntermediateArray ExpandDims(int axis)
    {
        if (axis < 0)
            axis += Shape.Length + 1; // allow negative indexing like NumPy

        if (axis < 0 || axis > Shape.Length)
            throw new ArgumentException("Axis out of range.");

        // Build new shape
        int[] newShape = new int[Shape.Length + 1];
        for (int i = 0, j = 0; i < newShape.Length; i++)
        {
            if (i == axis)
                newShape[i] = 1;
            else
                newShape[i] = Shape[j++];
        }

        // Data stays identical, only shape changes
        return new IntermediateArray((float[])InternalData.Clone(), newShape);
    }

    /// <summary>
    /// NumPy-like elementwise equality comparison.
    /// Returns a float NDArray with 1.0 where elements are equal, 0.0 otherwise.
    /// </summary>
    public static IntermediateArray Equal(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException("Shapes must match for elementwise comparison (broadcasting not implemented).");

        float[] resultData = new float[a.InternalData.Length];
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = a.InternalData[i] == b.InternalData[i] ? 1.0f : 0.0f;

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// NumPy-like elementwise multiplication with a boolean scalar.
    /// If flag is true, returns a copy of the input NDArray.
    /// If flag is false, returns all zeros with the same shape.
    /// </summary>
    public static IntermediateArray operator *(IntermediateArray a, bool flag)
    {
        float[] resultData = new float[a.InternalData.Length];

        if (flag)
        {
            // Copy data as-is
            for (int i = 0; i < resultData.Length; i++)
                resultData[i] = a.InternalData[i];
        }
        else
        {
            // Already initialized to zeros, no need to loop
        }

        return new IntermediateArray(resultData, a.Shape.ToArray());
    }

    /// <summary>
    /// NumPy-like variance (var) with axis and keepdims support.
    /// Default is population variance (ddof=0).
    /// Supports 1D and 2D NDArray.
    /// </summary>
    public IntermediateArray Var(int? axis = null, bool keepdims = false)
    {
        if (axis == null)
        {
            // Variance of all elements
            float mean = InternalData.Average();
            float sumSq = 0f;
            foreach (var x in InternalData)
                sumSq += (x - mean) * (x - mean);

            float variance = sumSq / InternalData.Length;

            if (keepdims)
            {
                // Return with same rank, but each axis = 1
                int[] newShape = Shape.Select(_ => 1).ToArray();
                return new IntermediateArray(new float[] { variance }, newShape);
            }
            else
            {
                return new IntermediateArray(new float[] { variance }, new int[] { 1 });
            }
        }
        else if (Shape.Length == 2)
        {
            int rows = Shape[0];
            int cols = Shape[1];

            if (axis == 0) // variance down columns
            {
                float[] result = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    float[] col = new float[rows];
                    for (int i = 0; i < rows; i++)
                        col[i] = InternalData[i * cols + j];

                    float mean = col.Average();
                    float sumSq = 0f;
                    foreach (var x in col)
                        sumSq += (x - mean) * (x - mean);
                    result[j] = sumSq / rows;
                }

                if (keepdims)
                    return new IntermediateArray(result, new int[] { 1, cols });
                else
                    return new IntermediateArray(result, new int[] { cols });
            }
            else if (axis == 1) // variance across rows
            {
                float[] result = new float[rows];
                for (int i = 0; i < rows; i++)
                {
                    float[] row = new float[cols];
                    for (int j = 0; j < cols; j++)
                        row[j] = InternalData[i * cols + j];

                    float mean = row.Average();
                    float sumSq = 0f;
                    foreach (var x in row)
                        sumSq += (x - mean) * (x - mean);
                    result[i] = sumSq / cols;
                }

                if (keepdims)
                    return new IntermediateArray(result, new int[] { rows, 1 });
                else
                    return new IntermediateArray(result, new int[] { rows });
            }
            else
            {
                throw new ArgumentException("Axis out of range for 2D NDArray.");
            }
        }
        else if (Shape.Length == 1)
        {
            // Simple 1D variance
            return Var(null, keepdims);
        }
        else
        {
            throw new NotImplementedException("Var(axis, keepdims) only implemented for 1D and 2D NDArrays.");
        }
    }

    /// <summary>
    /// NumPy-like reshape function. 
    /// Allows one dimension to be -1, which will be inferred automatically.
    /// </summary>
    public IntermediateArray Reshape(params int[] newShape)
    {
        // Compute the total number of elements in the current array
        int totalSize = Shape.Aggregate(1, (a, b) => a * b);

        // Handle -1 (infer dimension)
        int negativeOneCount = newShape.Count(d => d == -1);
        if (negativeOneCount > 1)
            throw new ArgumentException("Only one dimension can be -1.");

        if (negativeOneCount == 1)
        {
            int knownProduct = 1;
            foreach (var d in newShape)
            {
                if (d != -1) knownProduct *= d;
            }

            if (totalSize % knownProduct != 0)
                throw new ArgumentException("Cannot reshape array: incompatible shape.");

            int inferredDim = totalSize / knownProduct;
            newShape = newShape.Select(d => d == -1 ? inferredDim : d).ToArray();
        }

        // Verify that total size matches
        int newTotal = newShape.Aggregate(1, (a, b) => a * b);
        if (newTotal != totalSize)
            throw new ArgumentException("Cannot reshape array: total size mismatch.");

        // Return new NDArray with same Data but new shape
        return new IntermediateArray(InternalData, newShape);
    }

    /// <summary>
    /// NumPy-like concatenate function. Concatenates multiple NDArrays along the specified axis.
    /// If axis is null, arrays are flattened before concatenation.
    /// </summary>
    public static IntermediateArray Concatenate(IntermediateArray[] arrays, int? axis = null)
    {
        if (arrays == null || arrays.Length == 0)
            throw new ArgumentException("At least one NDArray is required.");

        // If only one array, return a copy
        if (arrays.Length == 1)
            return new IntermediateArray((float[])arrays[0].InternalData.Clone(), arrays[0].Shape.ToArray());

        int ndim = arrays[0].Shape.Length;

        // If axis is null → flatten all and concat as 1D
        if (axis == null)
        {
            var allData = arrays.SelectMany(a => a.InternalData).ToArray();
            return new IntermediateArray(allData, new int[] { allData.Length });
        }

        int ax = axis.Value;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim)
            throw new ArgumentException("Axis out of range.");

        // Validate shapes
        int[] baseShape = arrays[0].Shape.ToArray();
        for (int i = 1; i < arrays.Length; i++)
        {
            if (arrays[i].Shape.Length != ndim)
                throw new ArgumentException("All arrays must have the same number of dimensions.");

            for (int d = 0; d < ndim; d++)
            {
                if (d == ax) continue; // ignore concat axis
                if (arrays[i].Shape[d] != baseShape[d])
                    throw new ArgumentException("Shapes must match except along the concatenation axis.");
            }
        }

        // Compute new shape
        int newAxisSize = arrays.Sum(a => a.Shape[ax]);
        int[] newShape = baseShape.ToArray();
        newShape[ax] = newAxisSize;

        // Allocate result
        float[] newData = new float[arrays.Sum(a => a.InternalData.Length)];

        // Concatenate data along axis
        int[] strides = GetStrides(baseShape);
        int[] newStrides = GetStrides(newShape);

        int offset = 0;
        foreach (var arr in arrays)
        {
            Array.Copy(arr.InternalData, 0, newData, offset, arr.InternalData.Length);
            offset += arr.InternalData.Length;
        }

        // For simplicity, we copy raw flattened data in row-major order
        return new IntermediateArray(newData, newShape);
    }

    /// <summary>
    /// Helper: compute strides for a shape (row-major).
    /// </summary>
    private static int[] GetStrides(int[] shape)
    {
        int ndim = shape.Length;
        int[] strides = new int[ndim];
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// NumPy-like split function. Splits an NDArray into multiple sub-arrays along the given axis.
    /// Supports either number of splits (equal chunks) or explicit indices.
    /// </summary>
    public IntermediateArray[] Split(int sections, int axis = 0)
    {
        int ndim = Shape.Length;
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim)
            throw new ArgumentException("Axis out of range.");

        int size = Shape[axis];
        if (size % sections != 0)
            throw new ArgumentException("Array cannot be split evenly along this axis.");

        int chunk = size / sections;
        int[] indices = Enumerable.Range(1, sections).Select(i => i * chunk).ToArray();
        return Split(indices, axis);
    }

    /// <summary>
    /// NumPy-like split function with explicit indices_or_sections.
    /// Splits along the given axis at the specified indices.
    /// </summary>
    public IntermediateArray[] Split(int[] indices, int axis = 0)
    {
        int ndim = Shape.Length;
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis >= ndim)
            throw new ArgumentException("Axis out of range.");

        List<IntermediateArray> results = new List<IntermediateArray>();
        int start = 0;

        foreach (var end in indices.Concat(new[] { Shape[axis] }))
        {
            int[] newShape = Shape.ToArray();
            newShape[axis] = end - start;

            // Strides for row-major layout
            int[] strides = GetStrides(Shape);

            // Collect slice
            List<float> sliceData = new List<float>();
            int[] idx = new int[ndim];
            for (int flat = 0; flat < InternalData.Length; flat++)
            {
                int remainder = flat;
                for (int d = 0; d < ndim; d++)
                {
                    idx[d] = remainder / strides[d];
                    remainder %= strides[d];
                }

                if (idx[axis] >= start && idx[axis] < end)
                    sliceData.Add(InternalData[flat]);
            }

            results.Add(new IntermediateArray(sliceData.ToArray(), newShape));
            start = end;
        }

        return results.ToArray();
    }

    /// <summary>
    /// Returns a slice of the IntermediateArray along the first axis (like arr[index] in NumPy).
    /// </summary>
    public IntermediateArray IndexRow(int index)
    {
        if (Shape.Length < 1)
            throw new InvalidOperationException("Cannot index into a scalar IntermediateArray.");

        if (index < 0 || index >= Shape[0])
            throw new ArgumentOutOfRangeException(nameof(index), "Index out of range.");

        int ndim = Shape.Length;

        // Compute new shape (drop first axis)
        int[] newShape = Shape.Skip(1).ToArray();
        if (newShape.Length == 0)
            newShape = new int[] { 1 }; // special case → return scalar as 1-element IntermediateArray

        // Strides for row-major layout
        int[] strides = GetStrides(Shape);

        // Extract slice data
        List<float> sliceData = new List<float>();
        int[] idx = new int[ndim];
        for (int flat = 0; flat < InternalData.Length; flat++)
        {
            int remainder = flat;
            for (int d = 0; d < ndim; d++)
            {
                idx[d] = remainder / strides[d];
                remainder %= strides[d];
            }

            if (idx[0] == index)
                sliceData.Add(InternalData[flat]);
        }

        return new IntermediateArray(sliceData.ToArray(), newShape);
    }

    /// <summary>
    /// NumPy-like stack function. Stacks a sequence of IntermediateArrays along a new axis.
    /// </summary>
    public static IntermediateArray Stack(IntermediateArray[] arrays, int axis = 0)
    {
        if (arrays == null || arrays.Length == 0)
            throw new ArgumentException("Need at least one array to stack.");

        // All arrays must have the same shape
        int[] baseShape = arrays[0].Shape;
        foreach (var arr in arrays)
        {
            if (!arr.Shape.SequenceEqual(baseShape))
                throw new ArgumentException("All input arrays must have the same shape.");
        }

        int ndim = baseShape.Length + 1;
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis > ndim)
            throw new ArgumentException("Axis out of range.");

        // New shape: insert arrays.Length at axis
        int[] newShape = new int[ndim];
        for (int i = 0, j = 0; i < ndim; i++)
        {
            if (i == axis)
                newShape[i] = arrays.Length;
            else
                newShape[i] = baseShape[j++];
        }

        // Flatten and merge
        List<float> stacked = new List<float>();
        int[] strides = GetStrides(newShape);

        // Iterate in row-major order for new array
        int totalSize = newShape.Aggregate(1, (a, b) => a * b);
        int[] idx = new int[ndim];

        for (int flat = 0; flat < totalSize; flat++)
        {
            int remainder = flat;
            for (int d = 0; d < ndim; d++)
            {
                idx[d] = remainder / strides[d];
                remainder %= strides[d];
            }

            int arrIndex = idx[axis];
            int[] subIdx = idx.Where((_, d) => d != axis).ToArray();

            // Compute flat index in the source array
            int[] subStrides = GetStrides(baseShape);
            int subFlat = 0;
            for (int d = 0; d < subIdx.Length; d++)
                subFlat += subIdx[d] * subStrides[d];

            stacked.Add(arrays[arrIndex].InternalData[subFlat]);
        }

        return new IntermediateArray(stacked.ToArray(), newShape);
    }

    /// <summary>
    /// NumPy-like where function: elementwise selection between two NDArrays based on a boolean condition array.
    /// </summary>
    public static IntermediateArray Where(IntermediateArray condition, IntermediateArray x, IntermediateArray y)
    {
        if (condition == null || x == null || y == null)
            throw new ArgumentNullException("Arguments cannot be null.");

        if (!x.Shape.SequenceEqual(y.Shape))
            throw new ArgumentException("x and y must have the same shape.");

        if (!condition.Shape.SequenceEqual(x.Shape))
            throw new ArgumentException("Condition must have the same shape as x and y.");

        float[] result = new float[x.InternalData.Length];

        for (int i = 0; i < result.Length; i++)
        {
            // Treat nonzero as true
            bool cond = condition.InternalData[i] != 0.0f;
            result[i] = cond ? x.InternalData[i] : y.InternalData[i];
        }

        return new IntermediateArray(result, x.Shape);
    }

    /// <summary>
    /// NumPy-like where function: elementwise selection between an NDArray and a scalar float
    /// based on a boolean condition array.
    /// </summary>
    public static IntermediateArray Where(IntermediateArray condition, IntermediateArray x, float y)
    {
        if (condition == null || x == null)
            throw new ArgumentNullException("Arguments cannot be null.");

        if (!condition.Shape.SequenceEqual(x.Shape))
            throw new ArgumentException("Condition must have the same shape as x.");

        float[] result = new float[x.InternalData.Length];

        for (int i = 0; i < result.Length; i++)
        {
            // condition: nonzero = true
            bool cond = condition.InternalData[i] != 0.0f;
            result[i] = cond ? x.InternalData[i] : y;
        }

        return new IntermediateArray(result, x.Shape);
    }

    /// <summary>
    /// Sets part of an NDArray at a given index (row replacement for 2D arrays).
    /// Similar to NumPy: arr[index] = value
    /// </summary>
    public void SetIndex(int index, IntermediateArray value)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("SetIndex currently supports only 2D arrays.");

        int rows = Shape[0];
        int cols = Shape[1];

        if (index < 0 || index >= rows)
            throw new ArgumentOutOfRangeException(nameof(index), "Index out of bounds.");

        if (value.Shape.Length != 1 || value.Shape[0] != cols)
            throw new ArgumentException("Value must be a 1D array with the same number of columns.");

        // Find the starting offset in the flat Data
        int start = index * cols;

        for (int i = 0; i < cols; i++)
        {
            InternalData[start + i] = value.InternalData[i];
        }
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
}
