using System.Reflection.Metadata;
using AILibrary.Temp;

namespace AILibrary.Temp;

public static class Utils
{
    /// <summary>
    /// Creates new instance of the Tensor class.
    /// </summary>
    /// <param name="data">Iterable containing the data to be stored in the Tensor.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containing "data".</returns>
    public static Tensor tensor(object data, bool requires_grad = false)
    {
        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates a Parameter for your model (an instance of the Tensor class).
    /// </summary>
    /// <param name="data">Iterable containing the data to be stored in the Tensor.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containing "data".</returns>
    public static Parameter parameter(object data, bool requires_grad = false)
    {
        return new Parameter(data, requires_grad: true);
    }

    /// <summary>
    /// Creates new instance of the Tensor class, filled with zeros.
    /// </summary>
    /// <param name="shape">Iterable with the shape of the resulting Tensor.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining zeros with "shape" shape.</returns>
    public static Tensor zeros(dynamic shape, bool requires_grad = false)
    {
        var data = np.zeros(shape);
        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class, filled with ones.
    /// </summary>
    /// <param name="shape">Iterable with the shape of the resulting Tensor.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining ones with "shape" shape.</returns>
    public static Tensor ones(dynamic shape, bool requires_grad = false)
    {
        var data = np.zeros(shape);
        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class, filled with random integers.
    /// </summary>
    /// <param name="shape">iterable with the shape of the resulting Tensor.</param>
    /// <param name="low">lowest integer to be generated. [OPTIONAL]</param>
    /// <param name="high">one above the highest integer to be generated.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining random integers with "shape" shape.</returns>
    public static Tensor randint(dynamic shape, int low = 0, int high = -1, bool requires_grad = false)
    {
        dynamic data;

        if (high > low)
        {
            var data = np.random.randint(low, high, size: shape);
        }

        else 
        {
            var data = np.random.randint(low, size: shape);
        }

        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.
    /// </summary>
    /// <param name="shape">Iterable with the shape of the resulting Tensor.</param>
    /// <param name="xavier">Whether to use Xavier initialization on tensor (scale by squre root of first dimension).</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining normally distributed floats with "shape" shape.</returns>
    public static Tensor randn(dynamic shape, bool xavier = false, bool requires_grad = false)
    {
        var data = np.random.randn(*shape);

        if (xavier)
        {
            data /= NetPipeStyleUriParser.sqrt(shape[0]);
        }

        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.
    /// </summary>
    /// <param name="shape">Iterable with the shape of the resulting Tensor.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining normally distributed floats with "shape" shape.</returns>
    public static Tensor rand(dynamic shape, bool requires_grad = false)
    {
        var data = np.random.randn(shape);
        return new Tensor(data, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class with same shape as given Tensor, and filled with zeros.
    /// </summary>
    /// <param name="other">Tensor to copy shape from.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining zeros with other Tensor's shape.</returns>
    public static Tensor zeros_like(Tensor other, bool requires_grad = false)
    {
        var shape = other.shape;
        return zeros(shape: shape, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class with same shape as given Tensor, and filled with ones.
    /// </summary>
    /// <param name="other">Tensor to copy shape from.</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining ones with other Tensor's shape.</returns>
    public static Tensor ones_like(Tensor other, bool requires_grad = false)
    {
        var shape = other.shape;
        return ones(shape: shape, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class with same shape as given Tensor, and filled with random floats in a normal distribution.
    /// </summary>
    /// <param name="other">Tensor to copy shape from.</param>
    /// <param name="xavier">Whether to use Xavier initialization on tensor(scale by squre root of first dimension).</param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining normally distributed floats with other Tensor's shape.</returns>
    public static Tensor randn_like(Tensor other, bool xavier = true, bool requires_grad = false)
    {
        var shape = other.shape;
        return randn(shape: shape, xavier: xavier, requires_grad: requires_grad);
    }

    /// <summary>
    /// Creates new instance of the Tensor class with same shape as given Tensor, and filled with random integers in the given distribution.
    /// </summary>
    /// <param name="other">Tensor to copy shape from.</param>
    /// <param name="low"></param>
    /// <param name="high"></param>
    /// <param name="requires_grad">Whether to keep track of the Tensor's gradients.</param>
    /// <returns>Tensor containining normally distributed floats with other Tensor's shape.</returns>
    public static Tensor randint_like(Tensor other, int low, int high = 0, bool requires_grad = false)
    {
        var shape = other.shape;

        if (high == 0)
        {
            return randint(low, shape, requires_grad: requires_grad);
        }

        else
        {
            return randint(low, high, shape, requires_grad: requires_grad);
        }
    }

    /// <summary>
    /// Returns the largest values across the "dim" dimention. Example: (B, T, D), dim = 1-> (B, D).
    /// </summary>
    /// <param name="a">Tensor to perform the max() operation.</param>
    /// <param name="dim">Dimention to be reduced(only largest remains).</param>
    /// <param name="keepdims"></param>
    /// <returns>Whether to broadcast result to same shape as input.</returns>
    public static Tensor max(Tensor a, int dim = -1, bool keepdims = false)
    {
        return a.max(dim: dim, keepdims: keepdims);
    }

    /// <summary>
    /// Returns the index of the largest values across the "dim" dimention. Example: (B, T, D), dim = 1-> (B, D).
    /// </summary>
    /// <param name="a">Tensor to perform the argmax() operation.</param>
    /// <param name="dim">Dimention to be reduced(only largest index remains).</param>
    /// <param name="keepdims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the index of the largest values across the "dim" dimention.</returns>
    public static Tensor argmax(Tensor a, int dim = -1, bool keepdims = false)
    {
        return new Tensor(np.argmax(a._data, axis: dim, keepdims: keepdims));
    }

    /// <summary>
    /// Returns the sum of all values across the "dim" dimention. Example: (B, T, D), dim = 1-> (B, D).
    /// </summary>
    /// <param name="a">Tensor to perform the sum() operation.</param>
    /// <param name="dim">Dimention to be summed across.</param>
    /// <param name="keepdims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the sum of all values across the "dim" dimention.</returns>
    public static Tensor sum(Tensor a, int dim = -1, bool keepdims = false)
    {
        return a.sum(dim: dim, keepdims: keepdims);
    }

    /// <summary>
    /// Returns the mean of all values across the "dim" dimention. Example: (B, T, D), dim = 1-> (B, D).
    /// </summary> 
    /// <param name="a">Tensor to perform the mean() operation.</param>
    /// <param name="dim">Dimention to be averaged across.</param>
    /// <param name="keepdims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the mean of all values across the "dim" dimention.</returns>
    public static Tensor mean(Tensor a, int dim = -1, bool keepdims = false)
    {
        return a.mean(dim: dim, keepdims: keepdims);
    }

    /// <summary>
    /// Returns the variance of all values across the "dim" dimention. Example: (B, T, D), dim = 1-> (B, D).
    /// </summary>
    /// <param name="a">Tensor to perform the var() operation.</param>
    /// <param name="dim">Dimention the variance will be computed across.</param>
    /// <param name="keepdims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the variance of all values across the "dim" dimention.</returns>
    public static Tensor var(Tensor a, int dim = -1, bool keepdims = false)
    {
        return a.var(dim: dim, keepdims: keepdims);
    }

    /// <summary>
    /// Element-wise exponentiation of the "a" Tensor.
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    public static Tensor exp(Tensor a)
    {
        var op = new Exp();
        return op.forward(a);
    }

    /// <summary>
    /// Element-wise natural logarithm of the "a" Tensor.
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    public static Tensor log(Tensor a)
    {
        var op = new Log();
        return op.forward(a);
    }

    /// <summary>
    /// Element-wise square root of the "a" Tensor.
    /// </summary>
    /// <param name="a"></param>
    /// <returns></returns>
    public static Tensor sqrt(Tensor a)
    {
        var op = new Sqrt();
        return op.forward(a);
    }

    /// <summary>
    /// Returns the "a" tensor with all values where condition is True set to "value".
    /// </summary>
    /// <param name="condition">Two dimentions to be transposed.</param>
    /// <param name="a">Tensor to be filled by "value" where condition is True.</param>
    /// <param name="value">Value to fill Tensor with, where condition is True.</param>
    /// <returns>Returns the "a" tensor with all values where condition is True set to "value".</returns>
    public static Tensor where(object condition, Tensor a, float value)
    {
        return a.masked_fill(condition, value);
    }

    /// <summary>
    /// Returns the original tensor reshaped to the new shape given. Example: (16, 8, 4), *shape = (2, 32, 8)-> (2, 32, 8)
    /// </summary>
    /// <param name="a">Tensor to perform the reshape() operation.</param>
    /// <param name="shape">New shape of the tensor.</param>
    /// <returns>Returns the original tensor reshaped to the new shape given.</returns>
    public static Tensor reshape(Tensor a, dynamic shape)
    {
        return a.reshape(*shape);
    }

    /// <summary>
    /// Returns the original tensor with the two given dimentions transposed. Example: (16, 8, 4), *dims = (-2, -1)-> (16, 4, 8).
    /// </summary>
    /// <param name="a">Tensor to perform the transpose() operation.</param>
    /// <param name="dims">Two dimentions to be transposed.</param>
    /// <returns>Returns the original tensor with the two given dimentions transposed.</returns>
    public static Tensor transpose(Tensor a, dynamic dims)
    {
        return a.transpose(*dims);
    }

    /// <summary>
    /// Concatenates all tensors across an existing dimention. Example: [(B, T, D), (C, T, D)], dim = 0-> (B + C, T, D).
    /// </summary>
    /// <param name="tensors">Tensors to be concatenated.</param>
    /// <param name="dim"></param>
    /// <returns>Dimention to be concatenate across.</returns>
    public static Tensor cat(Tensor[] tensors, int dim)
    {
        var op = new Cat();
        return op.forward(tensors, dim);
    }

    /// <summary>
    /// Stacks all tensors across a new dimention. Example: [(B, T, D), (C, T, D)], dim = 0-> (2, B, T, D).
    /// </summary>
    /// <param name="tensors">Tensors to be stacked.</param>
    /// <param name="dim">Position of the new dimention to stack across.</param>
    /// <returns></returns>
    public static Tensor stack(Tensor[] tensors, int dim)
    {
        var op = new Stack();
        return op.forward(tensors, dim);
    }
}
