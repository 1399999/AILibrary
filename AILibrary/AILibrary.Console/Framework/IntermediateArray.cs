namespace AILibrary.Framework;

public class IntermediateArray
{
    public float[] InternalData { get; }
    public int[] Shape { get; }

    #region Constructor (Not Fully Broadcasted)

    public IntermediateArray(float[] data, int[] shape)
    {
        if (MultiplyTotal(shape) != data.Length)
        {
            throw new ArgumentException("Data length does not match shape.");
        }

        InternalData = data;
        Shape = shape;
    }

    // Generic constructor: Accepts any nested float arrays.
    public IntermediateArray(object array)
    {
        ArgumentNullException.ThrowIfNull(array);

        List<int> shapeList = new List<int>();
        List<float> flatData = new List<float>();

        ParseArray(array, shapeList, flatData, 0);

        Shape = shapeList.ToArray();
        InternalData = flatData.ToArray();
    }

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

    #endregion
    #region PEMDAS Operations (Not Fully Broadcasted)

    public static IntermediateArray Add(IntermediateArray a, IntermediateArray b) 
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            (a, b) = a.Broadcast(b);
        }

        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = a.InternalData[i] + b.InternalData[i];
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator +(IntermediateArray a, IntermediateArray b) => Add(a, b);

    public static IntermediateArray Subtract(IntermediateArray a, IntermediateArray b) 
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            (a, b) = a.Broadcast(b);
        }

        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = a.InternalData[i] - b.InternalData[i];
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator -(IntermediateArray a, IntermediateArray b) => Subtract(a, b);

    public IntermediateArray Negate()
    {
        float[] resultData = new float[InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = -InternalData[i];
        }

        return new IntermediateArray(resultData, Shape);
    }

    public static IntermediateArray operator -(IntermediateArray a) => a.Negate();

    public static IntermediateArray Multiply(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            (a, b) = a.Broadcast(b);
        }

        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = a.InternalData[i] * b.InternalData[i];
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator *(IntermediateArray a, IntermediateArray b) => Multiply(a, b);

    public static IntermediateArray operator *(IntermediateArray array, float scalar)
    {
        float[] resultData = new float[array.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = array.InternalData[i] * scalar;
        }

        return new IntermediateArray(resultData, array.Shape);
    }

    /// <summary>
    /// Elementwise multiplication with a boolean scalar.
    /// If flag is true, returns a copy of the input NDArray.
    /// If flag is false, returns all zeros with the same shape.
    /// </summary>
    public static IntermediateArray operator *(IntermediateArray a, bool flag)
    {
        float[] resultData = new float[a.InternalData.Length];

        if (flag)
        {
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.InternalData[i];
            }
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray Divide(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            (a, b) = a.BroadcastOther(b);
        }

        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = a.InternalData[i] / b.InternalData[i];
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator /(IntermediateArray array, float scalar)
    {
        float[] resultData = new float[array.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = array.InternalData[i] / scalar;
        }

        return new IntermediateArray(resultData, array.Shape);
    }

    public static IntermediateArray operator /(IntermediateArray a, IntermediateArray b) => Divide(a, b);

    public static IntermediateArray Power(IntermediateArray a, IntermediateArray b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            (a, b) = a.Broadcast(b);
        }

        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = (float)Math.Pow(a.InternalData[i], b.InternalData[i]);
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator ^(IntermediateArray a, IntermediateArray b) => Power(a, b);

    public static IntermediateArray Power(IntermediateArray a, float exponent)
    {
        float[] resultData = new float[a.InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = (float)Math.Pow(a.InternalData[i], exponent);
        }

        return new IntermediateArray(resultData, a.Shape);
    }

    public static IntermediateArray operator ^(IntermediateArray a, float exponent) => Power(a, exponent);

    #endregion
    #region Index Operations (Not Fully Broadcasted)

    /// <summary>
    /// Examples: Index[2, 7, 8] -> InternalData[(2 * 27 * 22) + (7 * 22) + (8 * 1)], the list size is: [32, 27, 22]
    /// </summary>
    /// <param name="indexes"></param>
    /// <returns></returns>
    private int FlattenIndex(int[] indexes)
    {
        List<int> grandIndexes = new();

        for (int i = 0; i < indexes.Length - 1; i++)
        {
            grandIndexes.Add(indexes[i]);

            int j = i + 1;

            for (int k = 0; k < Shape.Length - j; k++)
            {
                grandIndexes[i] *= Shape[j + k];
            }
        }

        return grandIndexes.Sum() + indexes[^1];
    }

    /// <summary>
    /// Examples: Index[2, 7, 8] -> InternalData[(2 * 27 * 22) + (7 * 22) + (8 * 1)], the list size is: [32, 27, 22]
    /// </summary>
    /// <param name="indexes"></param>
    /// <returns></returns>
    private long FlattenIndex(long[] indexes)
    {
        List<long> grandIndexes = new();

        for (int i = 0; i < indexes.Length - 1; i++)
        {
            grandIndexes.Add(indexes[i]);

            int j = i + 1;

            for (int k = 0; k < Shape.Length - j; k++)
            {
                grandIndexes[i] *= Shape[j + k];
            }
        }

        return grandIndexes.Sum() + indexes[^1];
    }

    /// <summary>
    /// Expands the index and allows for.
    /// </summary>
    /// <param name="indices"></param>
    /// <returns></returns>
    public float this[params int[] indices]
    {
        get => InternalData[FlattenIndex(indices)];
        set => InternalData[FlattenIndex(indices)] = value;
    }

    /// <summary>
    /// Reshape the array. 
    /// Allows one dimension to be -1, which will be inferred automatically.
    /// </summary>
    public IntermediateArray Reshape(params int[] newShape)
    {
        int totalSize = MultiplyTotal(Shape);
        int negativeOneCount = newShape.Count(d => d == -1);

        if (negativeOneCount > 1)
        {
            throw new ArgumentException("Only one dimension can be -1.");
        }

        if (negativeOneCount == 1)
        {
            int knownProduct = 1;
            int bufferedIndex = 0;

            for (int i = 0; i < newShape.Length; i++)
            {
                if (newShape[i] != -1)
                {
                    knownProduct *= newShape[i];
                }

                else
                {
                    bufferedIndex = i;
                }
            }

            newShape[bufferedIndex] = totalSize / knownProduct;
        }

        if (MultiplyTotal(newShape) != totalSize)
        {
            throw new ArgumentException("Cannot reshape array: total size mismatch.");
        }

        // Return new NDArray with same Data but new shape
        return new IntermediateArray(InternalData, newShape);
    }

    #endregion
    #region Index Expansions (Fully Unrefactored)

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

    #endregion
    #region Unilateral Functions (Fully Refactored)

    /// <summary>
    /// Elementwise exponential function.
    /// Returns a new NDArray where each element is e^x.
    /// </summary>
    public IntermediateArray Exp() // REFACTORED
    {
        float[] resultData = new float[InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = (float)Math.Exp(InternalData[i]);
        }

        return new IntermediateArray(resultData, Shape);
    }

    /// <summary>
    /// Elementwise natural logarithm (ln).
    /// Returns a new IntermediateArray where each element is log_e(x).
    /// </summary>
    public IntermediateArray Log() // REFACTORED
    {
        float[] resultData = new float[InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = MathF.Log(InternalData[i]);
        }

        return new IntermediateArray(resultData, Shape);
    }

    /// <summary>
    /// Elementwise square root.
    /// Returns a new IntermediateArray where each element is sqrt(x).
    /// </summary>
    public IntermediateArray Sqrt() // REFACTORED
    {
        float[] resultData = new float[InternalData.Length];

        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = MathF.Sqrt(InternalData[i]);
        }

        return new IntermediateArray(resultData, Shape);
    }

    #endregion
    #region Unrefactored Operations (Not fully Refactored)

    /// <summary>
    /// Sum of all elements.
    /// </summary>
    public IntermediateArray Sum()
    {
        float total = 0F;

        for (int i = 0; i < InternalData.Length; i++)
        {
            total += InternalData[i];
        }

        return new IntermediateArray(new float[] { total }, new int[] { });
    }

    /// <summary>
    /// Sum of one axis.
    /// </summary>
    public IntermediateArray Sum(int axis, bool keepdims = false)
    {
        if (Shape.Length <= axis)
        {
            throw new ArgumentException();
        }

        float[] output = new float[InternalData.Length / Shape[axis]];

        int totalIndex = 0;

        for (int i = 0; i < Shape[axis]; i++)
        {
            for (int j = 0; j < InternalData.Length / Shape[axis]; j++, totalIndex++)
            {
                output[j] += InternalData[totalIndex];
            }
        }

        return new IntermediateArray(output, !keepdims ? Shape.RemoveItem(Shape[axis]) : Shape.InsertItem(axis, 1));
    }

    /// <summary>
    /// Multiplies everything.
    /// </summary>
    public IntermediateArray Prod()
    {
        float product = 1F;

        foreach (var v in InternalData)
        {
            product *= v;
        }

        return new IntermediateArray(new float[] { product });
    }

    /// <summary>
    /// Product of one axis.
    /// </summary>
    public IntermediateArray Prod(int axis, bool keepdims = false)
    {
        if (Shape.Length <= axis)
        {
            throw new ArgumentException();
        }

        float[] output = new int[] { InternalData.Length / Shape[axis] }.OnesFloat();

        int totalIndex = 0;

        for (int i = 0; i < Shape[axis]; i++)
        {
            for (int j = 0; j < InternalData.Length / Shape[axis]; j++, totalIndex++)
            {
                output[j] *= InternalData[totalIndex];
            }
        }

        return new IntermediateArray(output, !keepdims ? Shape.RemoveItem(Shape[axis]) : Shape.InsertItem(axis, 1));
    }

    /// <summary>
    /// Matrix multiplication.
    /// Handles 1D, 2D, and batched ND arrays.
    /// </summary>
    public IntermediateArray Matmul(IntermediateArray other) // NOT REFACTORED
    {
        // --- Handle 1D @ 1D (dot product) ---
        if (Shape.Length == 1 && other.Shape.Length == 1)
        {
            if (Shape[0] != other.Shape[0])
                throw new ArgumentException("Shapes not aligned for 1D dot product.");
            float sum = 0;
            for (int i = 0; i < Shape[0]; i++)
                sum += this.InternalData[i] * other.InternalData[i];
            return new IntermediateArray(new float[] { sum }, new int[] { });
        }

        // --- Handle 2D @ 2D (matrix multiplication) ---
        if (Shape.Length == 2 && other.Shape.Length == 2)
        {
            int m = Shape[0];   // rows of A
            int k1 = Shape[1];  // cols of A
            int k2 = other.Shape[0]; // rows of B
            int n = other.Shape[1];  // cols of B

            if (k1 != k2)
                throw new ArgumentException("Shapes not aligned for matmul: " +
                                            $"({m},{k1}) x ({k2},{n})");

            float[] result = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < k1; k++)
                    {
                        sum += this.InternalData[i * k1 + k] * other.InternalData[k * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }

            return new IntermediateArray(result, new int[] { m, n });
        }

        // --- Handle N-D (batched matmul) ---
        // All dimensions except last two must broadcast
        if (Shape.Length >= 2 && other.Shape.Length >= 2)
        {
            int[] batchShapeA = Shape.Take(Shape.Length - 2).ToArray();
            int[] batchShapeB = other.Shape.Take(other.Shape.Length - 2).ToArray();
            int[] broadcastShape = BroadcastShapes(batchShapeA, batchShapeB);

            int m = Shape[Shape.Length - 2];
            int k1 = Shape[Shape.Length - 1];
            int k2 = other.Shape[other.Shape.Length - 2];
            int n = other.Shape[other.Shape.Length - 1];

            if (k1 != k2)
                throw new ArgumentException("Shapes not aligned for batched matmul.");

            int batchSize = broadcastShape.Aggregate(1, (a, b) => a * b);
            float[] result = new float[batchSize * m * n];

            for (int b = 0; b < batchSize; b++)
            {
                // For simplicity: assume arrays already broadcast-compatible
                int offsetA = b * m * k1;
                int offsetB = b * k2 * n;
                int offsetR = b * m * n;

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < k1; k++)
                        {
                            sum += this.InternalData[offsetA + i * k1 + k] *
                                   other.InternalData[offsetB + k * n + j];
                        }
                        result[offsetR + i * n + j] = sum;
                    }
                }
            }

            return new IntermediateArray(result, broadcastShape.Concat(new int[] { m, n }).ToArray());
        }

        throw new NotImplementedException("Matmul only implemented for 1D, 2D, and batched ND arrays.");
    }

    /// <summary>
    /// NumPy-like SwapAxes function. 
    /// Returns a new NDArray with the given axes swapped.
    /// </summary>
    public IntermediateArray SwapAxes(int axis1, int axis2) // NOT REFACTORED
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

    public IntermediateArray Mean()
    {
        // Global mean -> single scalar.
        float sum = 0F;

        for (int i = 0; i < InternalData.Length; i++)
        {
            sum += InternalData[i];
        }

        float mean = sum / InternalData.Length;

        return new IntermediateArray(new float[] { mean });
    }

    /// <summary>
    /// NumPy-like mean function.
    /// If axis is null, returns the mean of all elements (scalar IntermediateArray unless keepdims=true).
    /// If axis is specified, computes the mean along that axis.
    /// </summary>
    public IntermediateArray Mean(int axis, bool keepdims = false)
    {
        if (Shape.Length <= axis)
        {
            throw new ArgumentException();
        }

        float[] output = new float[InternalData.Length / Shape[axis]];

        int totalIndex = 0;

        for (int i = 0; i < Shape[axis]; i++)
        {
            for (int j = 0; j < InternalData.Length / Shape[axis]; j++, totalIndex++)
            {
                output[j] += InternalData[totalIndex];
            }
        }

        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= Shape[axis];
        }

        return new IntermediateArray(output, !keepdims ? Shape.RemoveItem(Shape[axis]) : Shape.InsertItem(axis, 1));
    }

    /// <summary>
    /// Maximum over all elements.
    /// </summary>
    public IntermediateArray Max()
    {
        float min = float.NegativeInfinity;

        for (int i = 0; i < InternalData.Length; i++)
        {
            if (InternalData[i] > min)
            {
                min = InternalData[i];
            }
        }

        return new IntermediateArray(new float[] { min });
    }

    /// <summary>
    /// Maximum along an axis.
    /// </summary>
    public IntermediateArray Max(int axis, bool keepdims = false)
    {
        if (Shape.Length <= axis)
        {
            throw new ArgumentException();
        }

        float[] output = float.NegativeInfinity.ValueFloat(new int[] { InternalData.Length / Shape[axis] });

        int totalIndex = 0;

        for (int i = 0; i < Shape[axis]; i++)
        {
            float temp = InternalData[i];

            for (int j = 0; j < InternalData.Length / Shape[axis]; j++, totalIndex++)
            {
                if (InternalData[totalIndex] > output[j])
                {
                    output[j] = InternalData[totalIndex];
                }
            }
        }

        return new IntermediateArray(output, !keepdims ? Shape.RemoveItem(Shape[axis]) : Shape.InsertItem(axis, 1));
    }

    /// <summary>
    /// Minimum over all elements.
    /// </summary>
    public IntermediateArray Min()
    {
        float min = float.PositiveInfinity;

        for (int i = 0; i < InternalData.Length; i++)
        {
            if (InternalData[i] < min)
            {
                min = InternalData[i];
            }
        }

        return new IntermediateArray(new float[] { min });
    }

    /// <summary>
    /// Minimum along an axis.
    /// </summary>
    public IntermediateArray Min(int axis, bool keepdims = false)
    {
        if (Shape.Length <= axis)
        {
            throw new ArgumentException();
        }

        float[] output = float.PositiveInfinity.ValueFloat(new int[] { InternalData.Length / Shape[axis] });

        int totalIndex = 0;

        for (int i = 0; i < Shape[axis]; i++)
        {
            float temp = InternalData[i];

            for (int j = 0; j < InternalData.Length / Shape[axis]; j++, totalIndex++)
            {
                if (InternalData[totalIndex] < output[j])
                {
                    output[j] = InternalData[totalIndex];
                }
            }
        }

        return new IntermediateArray(output, !keepdims ? Shape.RemoveItem(Shape[axis]) : Shape.InsertItem(axis, 1));
    }

    /// <summary>
    /// Calculates the variance of all elements.
    /// </summary>
    public IntermediateArray Var()
    {
        float mean = InternalData.Average();
        float sumSq = 0F;

        foreach (var x in InternalData)
        {
            sumSq += (x - mean) * (x - mean);
        }

        float variance = sumSq / InternalData.Length;

        return new IntermediateArray(new float[] { variance }, new int[] { 1 });
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
    /// NumPy-like advanced indexing assignment:
    /// Replaces rows selected by an index NDArray with the given value NDArray.
    /// Example: arr.SetIndex([0,2], [[10,10,10],[20,20,20]]).
    /// </summary>
    public void SetIndex(IntermediateArray index, IntermediateArray value)
    {
        if (Shape.Length < 2)
            throw new InvalidOperationException("SetIndex currently supports only 2D arrays.");

        if (index.Shape.Length != 1)
            throw new ArgumentException("Index must be a 1D NDArray of integers.");

        int rows = Shape[0];
        int cols = Shape[1];

        // Convert index.Data to integer indices
        int[] indices = index.InternalData.Select(f => (int)f).ToArray();

        foreach (var i in indices)
        {
            if (i < 0 || i >= rows)
                throw new ArgumentOutOfRangeException(nameof(index), "Index out of bounds.");
        }

        // Validate value shape
        if (value.Shape.Length == 1)
        {
            // A single row vector: broadcast across all indices
            if (value.Shape[0] != cols)
                throw new ArgumentException("1D value must match the number of columns.");

            for (int r = 0; r < indices.Length; r++)
            {
                int dstOffset = indices[r] * cols;
                Array.Copy(value.InternalData, 0, InternalData, dstOffset, cols);
            }
        }
        else if (value.Shape.Length == 2)
        {
            // Must match (len(indices), cols)
            if (value.Shape[0] != indices.Length || value.Shape[1] != cols)
                throw new ArgumentException("2D value must match (len(index), cols).");

            for (int r = 0; r < indices.Length; r++)
            {
                int dstOffset = indices[r] * cols;
                int srcOffset = r * cols;
                Array.Copy(value.InternalData, srcOffset, InternalData, dstOffset, cols);
            }
        }
        else
        {
            throw new ArgumentException("Value must be a 1D row or 2D array matching the target shape.");
        }
    }

    /// <summary>
    /// Returns a slice of the NDArray along the first axis (like arr[index] in NumPy).
    /// </summary>
    public IntermediateArray IndexRow(int index)
    {
        if (Shape.Length < 1)
            throw new InvalidOperationException("Cannot index into a scalar NDArray.");

        if (index < 0 || index >= Shape[0])
            throw new ArgumentOutOfRangeException(nameof(index), "Index out of range.");

        int ndim = Shape.Length;

        // Compute new shape (drop first axis)
        int[] newShape = Shape.Skip(1).ToArray();
        if (newShape.Length == 0)
            newShape = new int[] { 1 }; // special case → return scalar as 1-element NDArray

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
    /// Returns a slice of the IntermediateArray along the first axis (like arr[index] in NumPy).
    /// </summary>
    public IntermediateArray IndexRow(IntermediateArray values)
    {
        if (this.Shape.Length == values.Shape.Length && values.Shape.Length == 2)
        {
            float[][][] output = new float[this.Shape[0]][][];

            for (int i = 0; i < this.Shape[0]; i++)
            {
                output[i] = new float[this.Shape[1]][];

                for (int j = 0; j < this.Shape[1]; j++)
                {
                    output[i][j] = new float[values.Shape[1]];
                }
            }

            for (int i = 0; i < this.Shape[0]; i++)
            {
                //if (i >= values.Shape[0])
                //{
                //    break;
                //}

                for (int j = 0; j < this.Shape[1]; j++)
                {
                    for (int k = 0; k < values.Shape[1]; k++)
                    {
                        output[i][j][k] = values.InternalData[(((int)this.InternalData[(i * Shape[1]) + j]) * values.Shape[1]) + k];
                    }
                }
            }

            return new IntermediateArray(output);
        }

        bool isInt = true;
        int[] ints = new int[this.InternalData.Length];

        for (int i = 0; i < values.InternalData.Length; i++)
        {
            if (!int.TryParse(values.InternalData[i].ToString(), out int value))
            {
                isInt = false;
                ints[i] = value;
                break;
            }
        }

        if (isInt && values.Shape.Length == ints.Length)
        {
            return new IntermediateArray(GetMultiIndexData(ints));
        }

        throw new NotImplementedException();
    }

    //public static IntermediateArray IndexInto(IntermediateArray values, IntermediateArray indexers)
    //{

    //}

    #endregion
    #region Helper Functions (Fully Refactored)
    public float GetMultiIndexData(int[] indexes) => InternalData[FlattenIndex(indexes)];

    private static int MultiplyTotal(int[] indexes)
    {
        int output = 1;

        for (int i = 0; i < indexes.Length; i++)
        {
            output *= indexes[i];
        }

        return output;
    }

    public static IntermediateArray Value(float value, int[] shape) // REFACTORED
    {
        int expandedShape = MultiplyTotal(shape);
        float[] data = new float[expandedShape];

        for (int i = 0; i < expandedShape; i++)
        {
            data[i] = 1;
        }

        return new IntermediateArray(data, shape);
    }


    /// <summary>
    /// Example: Lengths [4,7,6]:
    /// Select: [2-3, 4-5, 0-1]
    /// </summary>
    /// <param name="arrays"></param>
    /// <returns>
    /// InternalData[(2*7*6)+(4*6)+0, (2*7*6)+(5*6)+0,
    /// (3*7*6)+(4*6)+0, (3*7*6)+(5*6)+0, 
    /// (2*7*6)+(4*6)+1, (2*7*6)+(5*6)+1, 
    /// (3*7*6)+(4*6)+1, (3*7*6)+(5*6)+1]
    /// </returns>
    /// <exception cref="Exception"></exception>
    public IntermediateArray Select(List<IntermediateArray> arrays) // REFACTORED
    {
        if (arrays.Count != Shape.Length)
        {
            throw new Exception();
        }

        long[][] indexes = new long[arrays[0].InternalData.Length][];

        for (int i = 0; i < arrays[0].InternalData.Length; i++)
        {
            indexes[i] = new long[arrays.Count];
        }

        for (int i = 0; i < arrays.Count; i++)
        {
            for (int j = 0; j < arrays[i].InternalData.Length; j++)
            {
                indexes[j][i] = (long)arrays[i][j];
            }
        }

        int[] kiloIndexes = new int[indexes.Length];

        for (int i = 0; i < indexes.Length; i++)
        {
            kiloIndexes[i] = (int)FlattenIndex(indexes[i]);
        }

        float[] output = new float[kiloIndexes.Length];

        for (int i = 0; i < kiloIndexes.Length; i++)
        {
            output[i] = InternalData[kiloIndexes[i]];
        }

        return new IntermediateArray(output);
    }

    public IntermediateArray ZerosLike() => new IntermediateArray(new float[MultiplyTotal(Shape)], Shape);
    public IntermediateArray OnesLike() => Value(1, Shape);

    #endregion
    #region Misscalenous Functions (Fully Unrefactored)

    /// <summary>
    /// Helper: Broadcast two shapes like NumPy
    /// </summary>
    private static int[] BroadcastShapes(int[] shapeA, int[] shapeB) // NOT REFACTORED
    {
        int ndim = Math.Max(shapeA.Length, shapeB.Length);
        int[] result = new int[ndim];

        for (int i = 0; i < ndim; i++)
        {
            int a = i < ndim - shapeA.Length ? 1 : shapeA[i - (ndim - shapeA.Length)];
            int b = i < ndim - shapeB.Length ? 1 : shapeB[i - (ndim - shapeB.Length)];

            if (a == b || a == 1 || b == 1)
                result[i] = Math.Max(a, b);
            else
                throw new ArgumentException("Shapes cannot be broadcast.");
        }
        return result;
    }

    /// <summary>
    /// Helper: compute strides for row-major order
    /// </summary>
    private static int[] ComputeStrides(int[] shape) // NOT REFACTORED
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

    #endregion
}
