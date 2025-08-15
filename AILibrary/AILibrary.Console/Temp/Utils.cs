using System;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AILibrary.NeuralNetworkFramework;
using static System.Formats.Asn1.AsnWriter;

namespace AILibrary.Temp;

public static class Utils
{
    public static Tensor tensor(object data, bool requires_grad = false)
    {
        /*
         Creates new instance of the Tensor class.

        @param data (Array-like): Iterable containing the data to be stored in the Tensor.
        @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

        @returns Tensor (Tensor): Tensor containing "data".
         */

        return new Tensor(data, requires_grad: requires_grad);
    }

    public static void parameter(object data, bool requires_grad = false)
    {
        /*
         Creates a Parameter for your model (an instance of the Tensor class).

        @param data (Array-like): Iterable containing the data to be stored in the Tensor.

        @returns Tensor (Tensor): Tensor containing "data".
         */

        return new Parameter(data, requires_grad: true);
    }

    public static Tensor zeros(dynamic shape, bool requires_grad = false)
    {
        /*
         Creates new instance of the Tensor class, filled with zeros.

        @param shape (tuple): iterable with the shape of the resulting Tensor.
        @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

        @returns Tensor (Tensor): Tensor containining zeros with "shape" shape.
         */
        var data = np.zeros(shape);

        return new Tensor(data, requires_grad: requires_grad);
    }

    public static Tensor ones(dynamic shape, bool requires_grad = false)
    {
        /*
         Creates new instance of the Tensor class, filled with ones.

        @param shape (tuple): iterable with the shape of the resulting Tensor.
        @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

        @returns Tensor (Tensor): Tensor containining ones with "shape" shape.
         */

        var data = np.zeros(shape);
        return new Tensor(data, requires_grad: requires_grad);
    }

    public static Tensor randint(dynamic shape, int low = 0, int high = -1, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class, filled with random integers.

        //@param low (int): lowest integer to be generated. [OPTIONAL]
        //@param high (int): one above the highest integer to be generated.
        //@param shape (tuple): iterable with the shape of the resulting Tensor.
        //@param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor (Tensor): Tensor containining random integers with "shape" shape.

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

    public static Tensor randn(dynamic shape, bool xavier = false, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.

        //@param shape(tuple): iterable with the shape of the resulting Tensor.
        //@param xavier (Bool): Whether to use Xavier initialization on tensor (scale by squre root of first dimension).
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining normally distributed floats with "shape" shape.

        var data = np.random.randn(*shape);

        if (xavier)
        {
            data /= NetPipeStyleUriParser.sqrt(shape[0]);
        }

        return new Tensor(data, requires_grad: requires_grad);
    }

    public static Tensor rand(dynamic shape, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.

        //@param shape(tuple): iterable with the shape of the resulting Tensor.
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining normally distributed floats with "shape" shape.

        var data = np.random.randn(shape);
        return new Tensor(data, requires_grad: requires_grad);
    }

    public static Tensor zeros_like(Tensor other, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class with same shape as given Tensor, and filled with zeros.
        //@param other(Tensor): Tensor to copy shape from.
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining zeros with other Tensor's shape.

        var shape = other.shape;
        return zeros(shape: shape, requires_grad: requires_grad);
    }

    public static Tensor ones_like(Tensor other, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class with same shape as given Tensor, and filled with ones.
        //@param other(Tensor): Tensor to copy shape from.
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining ones with other Tensor's shape.

        var shape = other.shape;
        return ones(shape: shape, requires_grad: requires_grad);
    }

    public static Tensor randn_like(Tensor other, bool xavier = true, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class with same shape as given Tensor,
        //and filled with random floats in a normal distribution.
        //@param other(Tensor): Tensor to copy shape from.
        //@param xavier(Bool): Whether to use Xavier initialization on tensor(scale by squre root of first dimension).
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining normally distributed floats with other Tensor's shape.

        var shape = other.shape;
        return randn(shape: shape, xavier: xavier, requires_grad: requires_grad);
    }

    public static Tensor randint_like(Tensor other, int low, int high = 0, bool requires_grad = false)
    {
        //Creates new instance of the Tensor class with same shape as given Tensor,
        //and filled with random integers in the given distribution.
        //@param other(Tensor): Tensor to copy shape from.
        //@param requires_grad(Bool): Whether to keep track of the Tensor's gradients.

        //@returns Tensor(Tensor): Tensor containining normally distributed floats with other Tensor's shape.

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

    public static Tensor max(Tensor a, int dim = -1, bool keepdims = false)
    {
        //Returns the largest values across the "dim" dimention.
        //Example: (B, T, D), dim = 1-> (B, D).

        //@param a(Tensor): tensor to perform the max() operation.
        //@param dim(int): dimention to be reduced(only largest remains).
        //@param keepdims(bool): wether to broadcast result to same shape as input.

        return a.max(dim: dim, keepdims: keepdims);
    }

    public static Tensor argmax(Tensor a, int dim = -1, bool keepdims = false)
    {
        //Returns the index of the largest values across the "dim" dimention.
        //Example: (B, T, D), dim = 1-> (B, D).

        //@param a(Tensor): tensor to perform the argmax() operation.
        //@param dim(int): dimention to be reduced(only largest index remains).
        //@param keepdims(bool): wether to broadcast result to same shape as input.

        return new Tensor(np.argmax(a._data, axis: dim, keepdims: keepdims));
    }

    public static Tensor sum(Tensor a, int dim = -1, bool keepdims = false)
    {
        //Returns the sum of all values across the "dim" dimention.
        //Example: (B, T, D), dim = 1-> (B, D).


        //@param a(Tensor): tensor to perform the sum() operation.
        //@param dim(int): dimention to be summed across.
        //@param keepdims(bool): wether to broadcast result to same shape as input.

        return a.sum(dim: dim, keepdims: keepdims);
    }

    public static Tensor mean(Tensor a, int dim = -1, bool keepdims = false)
    {
        //Returns the mean of all values across the "dim" dimention.
        //Example: (B, T, D), dim = 1-> (B, D).

        //@param a(Tensor): tensor to perform the mean() operation.
        //@param dim(int): dimention to be averaged across.
        //@param keepdims(bool): wether to broadcast result to same shape as input.

        return a.mean(dim: dim, keepdims: keepdims);
    }

    public static Tensor var(Tensor a, int dim = -1, bool keepdims = false)
    {
        //Returns the variance of all values across the "dim" dimention.
        //Example: (B, T, D), dim = 1-> (B, D).

        //@param a(Tensor): tensor to perform the var() operation.
        //@param dim(int): dimention the variance will be computed across.
        //@param keepdims(bool): wether to broadcast result to same shape as input.

        return a.var(dim: dim, keepdims: keepdims);
    }

    public static Tensor exp(Tensor a)
    {
        //Element-wise exponentiation of the "a" Tensor.

        var op = new Exp();
        return op.forward(a);
    }

    public static Tensor log(Tensor a)
    {
        //Element-wise natural logarithm of the "a" Tensor.

        var op = new Log();
        return op.forward(a);
    }

    public static Tensor sqrt(Tensor a)
    {
        //Element-wise square root of the "a" Tensor.

        var op = new Sqrt();
        return op.forward(a);
    }

    public static Tensor where(object condition, Tensor a, float value)
    {
        //Returns the "a" tensor with all values where condition is True set to "value".

        //@param condition(Array-like): two dimentions to be transposed.
        //@param a(Tensor): tensor to be filled by "value" where condition is True.
        //@param value (float): value to fill Tensor with, where condition is True.

        return a.masked_fill(condition, value);
    }

    public static Tensor reshape(Tensor a, dynamic shape)
    {
        //Returns the original tensor reshaped to the new shape given.
        //Example: (16, 8, 4), *shape = (2, 32, 8)-> (2, 32, 8)

        //@param a(Tensor): tensor to perform the reshape() operation.
        //@param* shape(integers): new shape of the tensor.

        return a.reshape(*shape);
    }

    public static Tensor transpose(Tensor a, dynamic dims)
    {
        //Returns the original tensor with the two given dimentions transposed.
        //Example: (16, 8, 4), *dims = (-2, -1)-> (16, 4, 8)

        //@param a(Tensor): tensor to perform the transpose() operation.
        //@param* dims(integers): two dimentions to be transposed.

        return a.transpose(*dims);
    }

    public static Tensor cat(Tensor[] tensors, int dim)
    {
        //Concatenates all tensors across an existing dimention.
        //Example: [(B, T, D), (C, T, D)], dim = 0-> (B + C, T, D).

        //@param tensors(list of Tensors): tensors to be concatenated.  
        //@param dim(int): dimention to be concatenate across.

        var op = new Cat();
        return op.forward(tensors, dim);
    }

    public static Tensor stack(Tensor[] tensors, int dim)
    {
        //Stacks all tensors across a new dimention.
        //Example: [(B, T, D), (C, T, D)], dim = 0-> (2, B, T, D).

        //@param tensors(list of Tensors): tensors to be stacked.  
        //@param dim(int): position of the new dimention to stack across.

        var op = new Stack();
        return op.forward(tensors, dim);
    }
}
