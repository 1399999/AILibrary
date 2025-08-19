using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AILibrary.Framework;

public class Tensor
{
    public IntermediateArray Data { get; private set; }
    public bool RequiresGrad { get; private set; }
    public dynamic? Operation { get; private set; }
    public List<Tensor> Children { get; private set; } = new List<Tensor>();
    public int[] Shape { get; private set; }
    public IntermediateArray? Grad { get; private set; }

    public Tensor(float data, bool requiresGrad = false, dynamic? operation = null)
    {
        Data = new IntermediateArray(new float[] { data }, new int[] { 1 });
        CtorCommon(requiresGrad, operation);
        Shape = new int[0];

        if (requiresGrad)
        {
            Grad = new IntermediateArray(new float[] { 0 }, new int[] { 1 });
        }
    }

    public Tensor(float[] data, bool requiresGrad = false, dynamic? operation = null)
    {
        Data = new IntermediateArray(data, new int[] { data.Length });
        CtorCommon(requiresGrad, operation);
    }

    public Tensor(float[][] data, bool requiresGrad = false, dynamic? operation = null)
    {
        Data = new IntermediateArray(data);
        CtorCommon(requiresGrad, operation);
    }

    public Tensor(float[][][] data, bool requiresGrad = false, dynamic? operation = null)
    {
        Data = new IntermediateArray(data);
        CtorCommon(requiresGrad, operation);
    }

    public Tensor(IntermediateArray data, bool requiresGrad = false, dynamic? operation = null)
    {
        Data = data;
        CtorCommon(requiresGrad, operation);
    }

    void CtorCommon(bool requiresGrad, dynamic? operation)
    {
        RequiresGrad = requiresGrad;
        Operation = operation;
        Shape = Data.Shape;

        if (requiresGrad)
        {
            Grad = Data.ZerosLike();
        }
    }

    public override string ToString()
    {
        return $"({Data}, requires_grad = {RequiresGrad})";
    }

    /// <summary>
    /// Performs the backpropagation with gradient descent from current tensor. Will fill every tensor's "grad" attribute with gradients relative to "self" (current Tensor).
    /// </summary>
    /// <param name="grad"></param>
    /// <param name="z"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public void Backward(IntermediateArray? grad = null, dynamic? z = null)
    {
        if (!RequiresGrad)
        {
            throw new InvalidOperationException();
        }

        if (grad == null)
        {
            grad = Data.OnesLike();
        }

        Grad = Grad + grad;

        if (z != null)
        {
            Children.Remove(z);
        }

        if (Operation != null)
        {
            if (Children.Count == 0)
            {
                Operation.Backward(Grad, this);
            }
        }
    }

    public void ToList()
    {
        throw new NotImplementedException();
    }

    public void ToArray()
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Reset the Tensor's gradients to zero.
    /// </summary>
    public void ZeroGrad()
    {
        Grad = Data.ZerosLike();
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
        dynamic op = new AddClass();
        return op.Forward(self, other);
    }

    public static Tensor operator +(Tensor self, float other)
    {
        dynamic op = new AddClass();
        return op.Forward(self, new Tensor(other));
    }

    /// <summary>
    /// New = self - other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator -(Tensor self, Tensor other) => self + other * -1;
    public static Tensor operator -(Tensor self, float other) => self + new Tensor(other) * -1;
    public static Tensor operator -(int self, Tensor other) 
    {
        if (self == 0)
        {
            dynamic op = new NegClass();
            return op.Forward(other);
        }

        throw new ArgumentException();
    }

    /// <summary>
    /// New = self * other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator *(Tensor self, Tensor other)
    {
        dynamic op = new MulClass();
        return op.Forward(self, other);
    }

    public static Tensor operator *(Tensor self, float other)
    {
        dynamic op = new MulClass();
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
        dynamic op = new PowClass();
        return op.Forward(self, new Tensor(other));
    }

    public static Tensor operator ^(Tensor self, Tensor other)
    {
        dynamic op = new PowClass();
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
        dynamic op = new MatMulClass();
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
        dynamic op = new DivClass();
        return op.Forward(self, other);
    }

    public static Tensor operator /(Tensor self, float other)
    {
        dynamic op = new DivClass();
        return op.Forward(self, other);
    }

    /// <summary>
    /// New = self[index]
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public Tensor IndexInto(Tensor index)
    {
        dynamic op = new SliceClass();
        return op.Forward(this, index);
    }

    /// <summary>
    /// Returns the largest values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be reduced (only largest remains).</param>
    /// <param name="keepDims">Whether to broadcast result to same shape as input.</param>
    /// <returns></returns>
    public Tensor GetMax(int dim = -1, bool keepDims = false)
    {
        dynamic op = new MaxClass();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the sum of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be summed across.</param>
    /// <param name="keepDims">Whether to broadcast result to same shape as input.</param>
    /// <returns>Returns the sum of all values across the "dim" dimention. </returns>
    public Tensor GetSum(int dim = -1, bool keepDims = false)
    {
        dynamic op = new SumClass();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the mean of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention to be averaged across.</param>
    /// <param name="keepDims">Wether to broadcast result to same shape as input.</param>
    /// <returns>Returns the mean of all values across the "dim" dimention.</returns>
    public Tensor GetMean(int dim = -1, bool keepDims = false)
    {
        dynamic op = new MeanClass();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the variance of all values across the "dim" dimention. Example: (B, T, D), dim = 1 -> (B, D).
    /// </summary>
    /// <param name="dim">Dimention the variance will be computed across</param>
    /// <param name="keepDims">Wether to broadcast result to same shape as input.</param>
    /// <returns>Returns the variance of all values across the "dim" dimention.</returns>
    public Tensor Var(int dim = -1, bool keepDims = false)
    {
        dynamic op = new VarClass();
        return op.Forward(this, dim, keepDims: keepDims);
    }

    /// <summary>
    /// Returns the original tensor reshaped to the new shape given. Example: (16, 8, 4), *shape =(2, 32, 8) -> (2, 32, 8).
    /// </summary>
    /// <param name="shape">Dimention the variance will be computed across.</param>
    /// <returns>Returns the original tensor reshaped to the new shape given.</returns>
    public Tensor Reshape(int[] shape)
    {
        dynamic op = new ReshapeClass();
        return op.Forward(this, shape);
    }

    /// <summary>
    /// Returns the original tensor with the two given dimentions transposed. Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8).
    /// </summary>
    /// <param name="dims">Two dimentions to be transposed.</param>
    /// <returns>Returns the original tensor with the two given dimentions transposed.</returns>
    public Tensor Transpose(int axis1, int axis2)
    {
        dynamic op = new TransposeClass();
        return op.Forward(this, axis1, axis2);
    }

    /// <summary>
    /// Returns the original tensor with the values where condition is True set to "value".
    /// </summary>
    /// <param name="condiditon">Matrix with True and False. Where this is False, will replace original with value.</param>
    /// <param name="value">Value to fill Tensor with, where condition is True.</param>
    /// <returns>Returns the original tensor with the values where condition is True set to "value".</returns>
    public Tensor MaskedFill(IntermediateArray condition, float value)
    {
        dynamic op = new MaskedFillClass();
        return op.Forward(this, condition, value);
    }

    private class AddClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            IntermediateArray data = a.Data + b.Data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new AddClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                var da = dz;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = a.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Length; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepdims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                var db = dz;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = b.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Length; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepdims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }

    private class NegClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a)
        {
            bool requiresGrad = a.RequiresGrad;

            // Get new Tensors's data:
            IntermediateArray data = -a.Data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new NegClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            a.Children.Add(z);
            Cache.Add(a);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];

            // Find gradients relative to "a", and pass it downstream:

            if (a.RequiresGrad) 
            {
                var da = -dz;
                a.Backward(da, z);
            }
        }
    }

    private class MulClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            IntermediateArray data = a.Data * b.Data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new MulClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(a*b) = b, apply chain rule:
                var da = dz * b.Data;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = a.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Length; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepdims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                // d/da(a*b) = a, apply chain rule:
                var db = dz * a.Data;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = b.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Length; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepdims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }

    private class DivClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            IntermediateArray data = a.Data / b.Data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new DivClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(a*b) = b, apply chain rule:
                var da = dz * (b.Data.OnesLike() / b.Data);

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = a.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Length; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepdims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                // d/da(a*b) = a, apply chain rule:
                var db = -dz * a.Data / (b.Data ^ 2);

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Length;
                int inDim = b.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Length; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepdims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }


    private class PowClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA, Tensor tensorB)
        {
            bool requiresGrad = tensorA.RequiresGrad;
            var data = tensorA.Data ^ tensorB.Data;

            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new PowClass());

            tensorA.Children.Add(z);

            Cache.Add(tensorA);
            Cache.Add(tensorB);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            var tensorA = Cache[0];
            var tensorB = Cache[1];

            if (tensorA.RequiresGrad)
            {
                var da = dz * (tensorB.Data * tensorA.Data ^ tensorB.Data - tensorB.Data.OnesLike());
                int gradDim = da.Shape.Length;
                int inDim = tensorA.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++) 
                {
                    da = da.Sum(axes: new int[] { 0 });
                }

                for (int n = 0; n < tensorA.Shape.Length; n++)
                {
                    if (tensorA.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepdims: true);
                    }
                }

                tensorA.Backward(da, z);
            }
        }
    }

    private class MatMulClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA, Tensor tensorB)
        {
            bool requiresGrad = tensorA.RequiresGrad || tensorB.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Matmul(tensorB.Data);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new MatMulClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            Parents.Add(tensorB);
            tensorA.Children.Add(z);
            tensorB.Children.Add(z);
            Cache.Add(tensorA);
            Cache.Add(tensorB);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Backprop through the matmul:
                var da = dz.Matmul(b.Data.SwapAxes(-1, -2));

                // Get difference between "a" size and upstream "da" size, to broadcast grad into "a":
                int gradDim = dz.Shape.Length;
                int inDim = a.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                // Backprop through the matmul:
                var db = a.Data.SwapAxes(-1, -2).Matmul(dz);

                // Get difference between "b" size and upstream "db" size, to broadcast grad into "b":
                int gradDim = dz.Shape.Length;
                int inDim = b.Shape.Length;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                b.Backward(db, z);
            }
        }
    }

    private class ExpClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public IntermediateArray? cacheExtension;

        public Tensor Forward(Tensor tensorA)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            IntermediateArray data = tensorA.Data.Exp();

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new ExpClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = data;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            IntermediateArray data = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(e^a) = e^a, apply the chain rule to the derivative of e^a:
                var da = data * dz;
                a.Backward(da, z);
            }
        }
    }

    private class LogClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Log();

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new LogClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(ln(a)) = (1/a), apply the chain rule to the derivative of the natural log:
                var da = a.Data.OnesLike() / a.Data * dz;
                a.Backward(da, z);
            }
        }
    }

    private class SqrtClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();
        public IntermediateArray? cacheExtension;

        public Tensor Forward(Tensor tensorA)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            IntermediateArray data = tensorA.Data.Sqrt();

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new SqrtClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = data;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            IntermediateArray data = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/dx(sqrt(a)) = (1/2) * (1/sqrt(a)), apply the chain rule to the derivative of the square root:
                var da = 1 / 2 * (data.OnesLike() / data) * dz;
                a.Backward(da, z);
            }
        }
    }

    private class SumClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA, int dim, bool keepdims)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Sum(axes: new int[1] { dim }, keepdims: keepdims);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new SumClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Expand upstream gradients to the shape of "a":
                var da = a.Data.OnesLike() * dz;
                a.Backward(da, z);
            }
        }
    }

    private class MeanClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int cacheExtension = 0;

        public Tensor Forward(Tensor tensorA, int dim, bool keepdims)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Mean(axis: dim);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new MeanClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = dim;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            int dim = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Propagate through the mean(x) operation:
                var da = a.Data.OnesLike() * dz;
                throw new NotImplementedException();
                //da /= a.Data[dim].Prod();
                a.Backward(da, z);
            }
        }
    }

    private class MaxClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int? cacheExtension;
        public IntermediateArray? cacheExtension2;

        public Tensor Forward(Tensor tensorA, int dim, bool keepdims = false)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            IntermediateArray data = tensorA.Data.Max(axis: dim, keepdims: keepdims);

            if (keepdims)
            {
                data = tensorA.Data.OnesLike() * data;
            }

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new MaxClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension2 = data;
            cacheExtension = dim;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            IntermediateArray? data = cacheExtension2;
            int? dim = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                var max = data;

                if (a.Shape != dz.Shape)
                {
                    // Broadcast upstream derivative to the size of "a":
                    dz = dz.ExpandDims((int) dim);
                    dz = dz * a.Data.OnesLike();

                    // Broadcast upstream output (max) to the size of "a":
                    max = data.ExpandDims((int)dim);
                    max = max * a.Data.OnesLike();
                }

                // Add upstream gradients to the [max] values:
                var da = dz * a.Data.Equals(max);

                a.Backward(da, z);
            }
        }
    }

    private class VarClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int cacheExtension;

        public Tensor Forward(Tensor tensorA, int dim, bool keepdims = false)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Var(axis: dim, keepdims: keepdims);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new VarClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = dim;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            int dim = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Propagate through the var(x) operation:

                IntermediateArray da = a.Data.OnesLike() * dz;
                throw new NotImplementedException();
                //da = da * 2 * (a.Data - a.Data.Mean(axis: dim, keepdims: true)) / np.array(a.Shape)[dim].Prod();

                //a.Backward(da, z);
            }
        }
    }

    private class ReshapeClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA, int[] shape)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.Reshape(shape);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new ReshapeClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Propagate through the var(x) operation:

                var da = dz.Reshape(a.Shape);

                a.Backward(da, z);
            }
        }
    }

    private class TransposeClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int[]? cacheExtension;

        public Tensor Forward(Tensor tensorA, int axis1, int axis2)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            var data = tensorA.Data.SwapAxes(axis1, axis2);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new TransposeClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = new int[] { axis1, axis2 };

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            int[]? dims = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Propagate through the var(x) operation:

                var da = dz.SwapAxes(dims[0], dims[1]);

                a.Backward(da, z);
            }
        }
    }

    private class CatClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int cacheExtension = 0;

        public Tensor Forward(List<Tensor> tensors, int dim)
        {
            bool requiresGrad = false;

            foreach (var tensor in tensors)
            {
                if (tensor.RequiresGrad)
                {
                    requiresGrad = true;
                }
            }

            // Get new Tensor's data:
            List<IntermediateArray> temp = new List<IntermediateArray>();

            for (int i = 0; i < tensors.Count; i++)
            {
                temp.Add(tensors[i].Data);
            }

            IntermediateArray data = IntermediateArray.Concatenate(temp.ToArray(), axis: dim);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new CatClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents = tensors;

            foreach (var tensor in tensors)
            {
                tensor.Children.Add(z);
            }

            Cache = tensors;
            cacheExtension = dim;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            List<Tensor> tensors = Cache;
            int dim = cacheExtension;

            dz = IntermediateArray.Concatenate(dz.Split(tensors.Count, dim));

            // Find gradients relative to each tensor in "tensor", and pass it downstream:
            for (int i = 0; i < tensors.Count; i++)
            {
                if (tensors[i].RequiresGrad)
                {
                    // For every tensor that generated the output, get gradients relative to that part of "dz": 
                    var di = dz.IndexRow(i);

                    tensors[i].Backward(di, z);
                }
            }
        }
    }

    private class StackClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int? cacheExtension;

        public Tensor Forward(List<Tensor> tensors, int dim)
        {
            // Verify if any original tensors requires grad:
            bool requiresGrad = false;

            foreach (var tensor in tensors)
            {
                if (tensor.RequiresGrad)
                {
                    requiresGrad = true;
                }
            }

            IntermediateArray[] temp = new IntermediateArray[tensors.Count];

            for (int i = 0; i < tensors.Count; i++)
            {
                temp[i] = tensors[i].Data;
            }

            // Get new Tensor's data:
            IntermediateArray data = IntermediateArray.Stack(temp, axis: dim);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new StackClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents = tensors;

            foreach (var tensor in tensors)
            {
                tensor.Children.Add(z);
            }

            Cache = tensors;
            cacheExtension = dim;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            List<Tensor> tensors = Cache;
            int? dim = cacheExtension;

            var dz2 = dz.Split(tensors.Count, axis: (int)dim);

            // Find gradients relative to each tensor in "tensor", and pass it downstream:
            for (int i = 0; i < tensors.Count; i++)
            {
                if (tensors[i].RequiresGrad)
                {
                    // For every tensor that generated the output, get gradients relative to that part of "dz": 
                    IntermediateArray di = dz2[i].IndexRow(i).Reshape(tensors[i].Data.Shape);

                    tensors[i].Backward(di, z);
                }
            }
        }
    }

    private class MaskedFillClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public dynamic? cacheExtension;

        public Tensor Forward(Tensor tensorA, IntermediateArray condition, float value)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            IntermediateArray data = IntermediateArray.Where(condition, tensorA.Data, value);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new MaskedFillClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = condition;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            dynamic? condition = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Because some activations are just set to a value, this operation is not differentiable.
                IntermediateArray da = IntermediateArray.Where(condition, dz, 0);

                a.Backward(da, z);
            }
        }
    }

    private class SliceClass()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public int cacheExtension = 0;

        public Tensor Forward(Tensor tensorA, int index)
        {
            bool requiresGrad = tensorA.RequiresGrad;

            // Get new Tensor's data:
            IntermediateArray data = tensorA.Data.IndexRow(index);

            // Create new Tensor:
            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new SliceClass());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(tensorA);
            tensorA.Children.Add(z);
            Cache.Add(tensorA);
            cacheExtension = index;

            return z;
        }

        public void Backward(IntermediateArray dz, Tensor z)
        {
            Tensor a = Cache[0];
            int index = cacheExtension;

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // Add upstream gradients to [index] part of da.
                IntermediateArray da = a.Data.ZerosLike();
                da.SetIndex(index, dz);

                a.Backward(da, z);
            }
        }
    }
}
