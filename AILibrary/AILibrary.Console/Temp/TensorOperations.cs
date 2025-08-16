namespace AILibrary.Temp;

public class Tensor
{
    public object _data { get; set; } // !!!!!!!!!!!!!!!!!!!
    public bool RequiresGrad { get; set; }
    public dynamic? Operation { get; set; }
    public dynamic Children { get; set; } // !!!!!!!!!!!!!!!!!!!
    public List<int> Shape { get; set; }
    public object? Grad { get; set; }

    public Tensor(object data, bool requiresGrad = false, dynamic? operation = null) // !!!!!!!!!!!!!!!!!!!
    {
        _data = data;
        RequiresGrad = requiresGrad;
        Operation = operation;
        Children = new List<Tensor>(); // !!!!!!!!!!!!!!!!!!!
        Shape = new List<int>();

        if (requiresGrad)
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

    public static Tensor operator +(Tensor self, float other)
    {
        dynamic op = new Add();
        return op.Forward(self, new Tensor(other));
    }

    /// <summary>
    /// New = self - other
    /// </summary>
    /// <param name="self"></param>
    /// <param name="other"></param>
    /// <returns></returns>
    public static Tensor operator -(Tensor self, Tensor other) => self + (other * -1);
    public static Tensor operator -(Tensor self, float other) => self + (new Tensor(other) * -1);
    public static Tensor operator -(int self, Tensor other) 
    {
        if (self == 0)
        {
            dynamic op = new Neg();
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

    \public static Tensor operator /(Tensor self, float other)
    {
        dynamic op = new Div();
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
        dynamic op = new Slice();
        return op.Forward(this, index);
    }

    /// <summary>
    /// New = self > other
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

    private class Add()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            object data = a._data + b._data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new Add());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(Tensor dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                var da = dz;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = a.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Count; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepDims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                var db = dz;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = b.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Count; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepDims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }

    private class Neg()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a)
        {
            bool requiresGrad = a.RequiresGrad;

            // Get new Tensors's data:
            object data = 0 - a._data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new Neg());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            a.Children.Add(z);
            Cache.Add(a);

            return z;
        }

        public void Backward(Tensor dz, Tensor z)
        {
            Tensor a = Cache[0];

            // Find gradients relative to "a", and pass it downstream:

            if (a.RequiresGrad) 
            {
                var da = 0 - dz;
                a.Backward(da, z);
            }
        }
    }

    private class Mul()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            object data = a._data * b._data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new Mul());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(Tensor dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(a*b) = b, apply chain rule:
                var da = dz * b._data;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = a.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Count; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepDims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                // d/da(a*b) = a, apply chain rule:
                var db = dz * a._data;

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = b.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Count; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepDims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }

    private class Div()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor a, Tensor b)
        {
            bool requiresGrad = a.RequiresGrad || b.RequiresGrad;

            // Get new Tensors's data:
            object data = a._data / b._data;

            // Create new Tensor's data:
            Tensor z = new Tensor(data, requiresGrad, operation: new Div());

            // Add new Tensors to "children" and old Tensors to "parents":
            Parents.Add(a);
            Parents.Add(b);
            a.Children.Add(z);
            b.Children.Add(z);
            Cache.Add(a);
            Cache.Add(b);

            return z;
        }

        public void Backward(Tensor dz, Tensor z)
        {
            Tensor a = Cache[0];
            Tensor b = Cache[1];

            // Find gradients relative to "a", and pass it downstream:
            if (a.RequiresGrad)
            {
                // d/da(a*b) = b, apply chain rule:
                var da = dz * (1 / b._data);

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = a.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    da = da.Sum(dim: 0);
                }

                for (int n = 0; n < a.Shape.Count; n++)
                {
                    if (a.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepDims: true);
                    }
                }

                a.Backward(da, z);
            }

            // Find gradients relative to "b", and pass it downstream:
            if (b.RequiresGrad)
            {
                // d/da(a*b) = a, apply chain rule:
                var db = -dz * a._data / (b._data ^ 2);

                // Rescale gradient to have the same shape "a":
                int gradDim = dz.Shape.Count;
                int inDim = b.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++)
                {
                    db = db.Sum(dim: 0);
                }

                for (int n = 0; n < b.Shape.Count; n++)
                {
                    if (b.Shape[n] == 1)
                    {
                        db = db.Sum(dim: n, keepDims: true);
                    }
                }

                b.Backward(db, z);
            }
        }
    }


    private class Pow()
    {
        public List<Tensor> Parents { get; set; } = new List<Tensor>();
        public List<Tensor> Cache { get; set; } = new List<Tensor>();

        public Tensor Forward(Tensor tensorA, Tensor tensorB)
        {
            bool requiresGrad = tensorA.RequiresGrad;
            var data = tensorA._data ^ tensorB._data;

            var z = new Tensor(data, requiresGrad: requiresGrad, operation: new Pow());

            tensorA.Children.Add(z);

            Cache.Add(tensorA);
            Cache.Add(tensorB);

            return z;
        }

        public void Backward(Tensor dz, Tensor z)
        {
            var tensorA = Cache[0];
            var tensorB = Cache[1];

            if (tensorA.RequiresGrad)
            {
                var da = dz * (tensorB._data * tensorA._data ^ (tensorB._data - 1));
                int gradDim = da.Shape.Count;
                int inDim = tensorA.Shape.Count;

                for (int i = 0; i < gradDim - inDim; i++) 
                {
                    da = da.Sum(axis: 0);
                }

                for (int n = 0; n < tensorA.Shape.Count; n++)
                {
                    if (tensorA.Shape[n] == 1)
                    {
                        da = da.Sum(dim: n, keepDims: true);
                    }
                }

                tensorA.Backward(da, z);
            }
        }
    }
}

class Parameter : Tensor
{
    public Parameter(object data, bool requires_grad = false, dynamic? operation = null) : base(data, requires_grad, operation)
    {

    }
}
