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
        var negTensor = -tensor;

        var expx = tensor.Exp();
        var negexpx = negTensor.Exp();

        var a = expx - negexpx;
        var b = expx + negexpx;

        return a / b;
    }

    //public static Tensor CrossEntropy(this Tensor logits, IntermediateArray labels)
    //{
    //    // Logits must be a 2D array, shape (batch, classes).
    //    // Example: (2, 3) for 2 samples and 3 possible classes.
    //        if (logits.Shape.Length != 2)
    //        throw new ArgumentException("CrossEntropy expects 2D logits of shape (batch_size, num_classes).");

    //    int batch = logits.Shape[0];
    //    int classes = logits.Shape[1];

    //    // --- Softmax computation with numerical stability ---
    //    //float[] stabilized = new float[logits.Data.InternalData.Length];

    //    // For each row, we subtract the maximum logit before exponentiation.
    //    //for (int i = 0; i < batch; i++)
    //    //{
    //    //    float maxVal = float.MinValue;
    //    //    for (int j = 0; j < classes; j++)
    //    //    {
    //    //        maxVal = Math.Max(maxVal, logits.Data[i * classes + j]);
    //    //    }

    //    //    for (int j = 0; j < classes; j++)
    //    //    {
    //    //        stabilized[i * classes + j] = logits.Data[i * classes + j] - maxVal;
    //    //        //logits.Data[i * classes + j] - maxVal;
    //    //    }
    //    //}

    //    //var stabilizedND = new Tensor(new IntermediateArray(stabilized, logits.Shape), operation: logits.Operation);

    //    float maxVal = float.MinValue;

    //    for (int i = 0; i < batch; i++)
    //    {
    //        for (int j = 0; j < classes; j++)
    //        {
    //            maxVal = Math.Max(maxVal, logits.Data[i * classes + j]);
    //        }
    //    }

    //    var stabilizedND = logits - maxVal;

    //    var expVals = stabilizedND.Exp(); // exp(logits)
    //    var expSums = expVals.Sum(1, keepdims: true); // sum(exp) per row
    //    var softmax = expVals / expSums; // Softmax probabilities
    //    var logSoftmax = softmax.IndexInto(ArangeInt(batch)).Log();

    //    float loss = 0f;

    //    // --- Case 1: One-hot labels ---
    //    if (labels.Shape.Length == 2)
    //    {
    //        if (labels.Shape[0] != batch || labels.Shape[1] != classes)
    //        {
    //            throw new ArgumentException("When labels are one-hot, they must have the same shape as logits.");
    //        }

    //        var mul = new Tensor(labels) * logSoftmax;
    //        var summed = mul.Sum(dim: -1, keepdims: false); // sum over classes
    //        return -summed.Sum() / batch;
    //    }
    //    // --- Case 2: Class index labels ---
    //    else if (labels.Shape.Length == 1)
    //    {
    //        if (labels.Shape[0] != batch)
    //        {
    //            throw new ArgumentException("When labels are class indices, they must have shape (batch,).");
    //        }

    //        for (int i = 0; i < batch; i++)
    //        {
    //            int classIdx = (int)labels.InternalData[i];

    //            if (classIdx < 0 || classIdx >= classes)
    //            {
    //                throw new ArgumentException("Label out of range of class count.");
    //            }

    //            float logProb = logSoftmax.Data.InternalData[i * classes + classIdx];
    //            loss += -logProb;
    //        }

    //        loss /= batch;

    //        //return logSoftmax.Mean();
    //    }
    //    else
    //    {
    //        throw new ArgumentException("Labels must be either one-hot encoded (2D) or class indices (1D).");
    //    }

    //    return new Tensor(new IntermediateArray(new float[] { loss }, new int[] { }));
    //}

    //public static Tensor CrossEntropy(this Tensor logits, IntermediateArray labels)
    //{
    //    var b = logits.Shape[0];
    //    var d = logits.Shape[1];
    //    logits = logits.Reshape([b, d]);

    //    logits = logits.Exp();
    //    logits = logits / logits.Sum(dim: 1, keepdims: true);

    //    labels = labels.Reshape([b]);

    //    var logLosses = logits.IndexInto(ArangeInt(b)).IndexInto(labels);
    //    var loss = -logLosses.Sum() / b;
    //    return loss;
    //}

    /// <summary>
    /// Cross Entropy Loss implementation (like PyTorch).
    /// Supports:
    /// - labels as one-hot vectors (batch, num_classes)
    /// - labels as class indices (batch,)
    /// Returns a scalar Tensor (mean loss).
    /// </summary>
    //public static Tensor CrossEntropy(this Tensor logits, Tensor labels)
    //{
    //    if (logits.Shape.Length != 2)
    //    {
    //        throw new ArgumentException("CrossEntropy expects 2D logits of shape (batch, num_classes).");
    //    }

    //    int batch = logits.Shape[0];
    //    int classes = logits.Shape[1];

    //    // --- 1. Stabilize logits (subtract max per row for numerical stability)
    //    var max = logits.ToItem().Max(); // (batch, 1)
    //    var stabilized = logits - max;               // broadcasted subtraction

    //    // --- 2. Compute softmax + log softmax
    //    var expVals = stabilized.Exp();
    //    var expSums = expVals.Sum(dim: 1, keepdims: true);
    //    var softmax = expVals / expSums;
    //    var logSoftmax = softmax.Log(); // (batch, classes)

    //    Tensor loss;

    //    // --- 3a. One-hot label case
    //    if (labels.Shape.Length == 2)
    //    {
    //        var mul = labels * logSoftmax;                // (batch, classes)
    //        var perSample = mul.Sum(dim: 1);              // (batch,)
    //        loss = -(perSample.Mean());                   // scalar
    //    }
    //    // --- 3b. Index label case (class indices per sample)
    //    else if (labels.Shape.Length == 1)
    //    {
    //        var perSample = new Tensor(new float[batch]);

    //        for (int i = 0; i < batch; i++)
    //        {
    //            int classIdx = (int)labels.IndexInto(new IntermediateArray(new float[] { i })).ToItem()[0]; // Grab label as int.
    //            float logProb = logSoftmax.IndexInto(new IntermediateArray(new float[] { i, classIdx })).ToItem()[0];
    //            perSample.Data.InternalData[i] = -logProb;
    //        }

    //        loss = perSample.Mean(); // scalar
    //    }
    //    else
    //    {
    //        throw new ArgumentException("Labels must be either (batch,) or (batch, num_classes).");
    //    }

    //    return loss;
    //}

    public static Tensor CrossEntropy(this Tensor logits, IntermediateArray labels)
    {
        var counts = logits.Exp();
        var prob = counts / counts.Sum(0, keepdims: true);

        var abcd = prob[[ArangeInt(32), labels]];
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
}
