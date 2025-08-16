using System.Drawing;
using System.Numerics;

namespace AILibrary.Temp;

public class Tensor
{
    public object _data { get; set; } // !!!!!!!!!!!!!!!!!!!
    public bool RequiresGrad { get; set; }
    public dynamic? Operation { get; set; }
    public dynamic Children { get; set; } // !!!!!!!!!!!!!!!!!!!
    public dynamic Shape { get; set; } // !!!!!!!!!!!!!!!!!!!
    public object Grad { get; set; }

    public Tensor(object data, bool requires_grad = false, dynamic? operation = null) // !!!!!!!!!!!!!!!!!!!
    {
        _data = data;
        RequiresGrad = requires_grad;
        Operation = operation;
        Children = new List<Tensor>(); // !!!!!!!!!!!!!!!!!!!
        Shape = 0; // !!!!!!!!!!!!!!!!!!!

        if (requires_grad)
        {
            Grad = Np.zeros_like(data);
        }
    }

    public override string ToString()
    {
        return $"({_data}, requires_grad = {RequiresGrad})";
    }

    /// <summary>
    /// Returns the data stored in the tensor as a Numpy Array.
    /// </summary>
    /// <returns></returns>
    public object Data()
    {
        return _data;
    }

    /// <summary>
    /// Performs the backpropagation with gradient descent from current tensor. Will fill every tensor's "grad" attribute with gradients relative to "self" (current Tensor).
    /// </summary>
    /// <param name="grad"></param>
    /// <param name="z"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public void Backward(dynamic? grad = null, dynamic? z = null)
    {
        if (!RequiresGrad)
        {
            throw new InvalidOperationException();
        }

        if (grad == null)
        {
            grad = np.ones_like(_data);
        }

        Grad += grad;

        if (z != null)
        {
            Children.Remove(z);
        }

        if (Operation != null)
        {
            if (Children.Length == 0)
            {
                Operation.Backward(Grad, this);
            }
        }
    }

    //public object ToList()
    //{

    //}

    //public object ToArray()
    //{

    //}

    /// <summary>
    /// ''' Reset the Tensor's gradients to zero. '''
    /// </summary>
    public void ZeroGrad()
    {
        Grad = Np.zeros_like(_data);
    }

    /// <summary>
    /// Reset the gradients of this Tensor, and of all of the Tensors that led to it.
    /// </summary>
    public void ZeroGradTree()
    {
        ZeroGrad();

        if (Operation != null)
        {
            foreach (var parent in Operation.Parents)
            {
                parent.ZeroGradTree();
            }

            Operation = null;
        }
    }

    /// <summary>
    /// New = self + other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator +(Tensor self, Tensor other)
    {
        dynamic op = new Add();
        return op.Forward(self, other);
    }

    /// <summary>
    /// New = self - other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator -(Tensor self, Tensor other) => self + (other * -1);

    /// <summary>
    /// New = self * other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator *(Tensor self, Tensor other)
    {
        dynamic op = new Mul();
        return op.Forward(self, other);
    }

    public static Tensor operator *(Tensor self, float other)
    {
        dynamic op = new Mul();
        return op.Forward(self, new Tensor(other));
    }

    /// <summary>
    /// New = self ^ other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator ^(Tensor self, float other)
    {
        dynamic op = new Pow();
        return op.Forward(self, new Tensor(other));
    }

    public static Tensor operator ^(Tensor self, Tensor other)
    {
        dynamic op = new Pow();
        return op.Forward(self, other);
    }

    /// <summary>
    /// New = self @ other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public Tensor Matmul(Tensor other)
    {
        dynamic op = new MatMul();
        return op.Forward(this, other);
    }

    /// <summary>
    /// New = self @ other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator /(Tensor self, Tensor other)
    {
        dynamic op = new Div();
        return op.Forward(self, other);
    }

    /// <summary>
    /// New = self > other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public Tensor IndexInto(Tensor index)
    {
        dynamic op = new Slice();
        return op.Forward(this, index);
    }

    /// <summary>
    /// New = self[index]
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor Gt(Tensor self, Tensor other)
    {
        return _data > other.ToArray();
    }

    /// <summary>
    /// Returns the largest values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be reduced (only largest remains).</param>
    /// <param name="keepDims">Whether to broadcast result to same shape as input.</param>
    /// <returns></returns>
    public Tensor Max(int dim = -1, bool keepDims = false)
    {
        dynamic op = new Max();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the sum of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be summed across.</param>
    /// <param name="keepDims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the sum of all values across the "dim" dimention. </returns>
    public Tensor Sum(int dim = -1, bool keepDims = false)
    {
        dynamic op = new Sum();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the mean of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be averaged across.</param>
    /// <param name="keepDims">Wether to broadcast result to same shape as input.</param>
    /// <returns>Returns the mean of all values across the "dim" dimention.</returns>
    public Tensor Mean(int dim = -1, bool keepDims = false)
    {
        dynamic op = new Mean();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the variance of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention the variance will be computed across</param>
    /// <param name="keepDims">Wether to broadcast result to same shape as input.</param>
    /// <returns>Returns the variance of all values across the "dim" dimention.</returns>
    public Tensor Vari(int dim = -1, bool keepDims = false)
    {
        dynamic op = new Var();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the original tensor reshaped to the new shape given. Example: (16, 8, 4), *shape =(2, 32, 8) -> (2, 32, 8).
    /// </summary>
    /// <param name="shape">Dimention the variance will be computed across.</param>
    /// <returns>Returns the original tensor reshaped to the new shape given.</returns>
    public Tensor Reshapei(dynamic shape) // !!!!!!!!!!!!!!
    {
        dynamic op = new Reshape();
        return op.Forward(this, shape);
    }

    /// <summary>
    /// Returns the original tensor with the two given dimentions transposed. Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8).
    /// </summary>
    /// <param name="dims">Two dimentions to be transposed.</param>
    /// <returns>Returns the original tensor with the two given dimentions transposed.</returns>
    public Tensor Transposei(dynamic dims) // !!!!!!!!!!!!!!
    {
        dynamic op = new Transpose();
        return op.Forward(this, dims);
    }

    /// <summary>
    /// Returns the original tensor with the values where condition is True set to "value".
    /// </summary>
    /// <param name="condiditon">Matrix with True and False. Where this is False, will replace original with value.</param>
    /// <param name="value">Value to fill Tensor with, where condition is True.</param>
    /// <returns>Returns the original tensor with the values where condition is True set to "value".</returns>
    public Tensor MaskedFilli(dynamic condition, float value) // !!!!!!!!!!!!!!
    {
        dynamic op = new MaskedFill();
        return op.Forward(this, condition, value);
    }
}

class Parameter : Tensor
{
    public Parameter(object data, bool requires_grad = false, dynamic? operation = null) : base(data, requires_grad, operation)
    {

    }
}
