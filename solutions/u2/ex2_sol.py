import timeit
import torch

torch.manual_seed(0)


# Note: this is just one of many possible solutions which enable
# backpropagation.
def mv(M, v):
    """Multiply the tensor matrix M by the tensor vector v.

    Args:
    * M: matrix of shape [n, m]
    * v: vector of shape [m]

    Result: vector of shape [n]
    """
    # Some assertions concerning the shapes of the arguments
    assert M.dim() == 2
    assert v.dim() == 1
    assert M.shape[1] == v.shape[0]

    # The output vector, of length n
    w = torch.zeros(M.shape[0])

    # Calculate the product
    for i, row in enumerate(M):
        # Use dot product, equivalent to `torch.sum(row * v)`
        w[i] = torch.dot(row, v)

    # Return the resulting vector
    return w


# Some checks to see if mv behaves as torch.mv
M = torch.randn([4, 2], dtype=torch.float, requires_grad=True)
v = torch.randn([2], dtype=torch.float, requires_grad=True)
assert all(torch.mv(M, v) == mv(M, v))

# Gradient computation using torch.mv
torch.mv(M, v).sum().backward()
print(v.grad)
# tensor([ -1.3190 , -0.2856])

# Zero-out the gradients
M.grad.zero_()
v.grad.zero_()
print(v.grad)
# tensor ([0. , 0.])

# Gradient computation using mv; the resulting gradient
# should be the same as with torch.mv
mv(M, v).sum().backward()
print(v.grad)
# tensor ([ -1.3190 , -0.2856])


##############################################################################
# Why we should not necessarily use a custom implementation of matrix-dot
# product in real code?
#
# The following were remarked:
#
# * Implementation of `mv` (forward pass) in PyTorch is significantly faster
# * For larger matrices/vectors, the gradients calculated with `torch.mv`
#   and custom `mv` are slightly different
#
# The main reason is probably that `torch.mv` is implemented in C/C++ and heavily
# optimized.  
#
# However, there's another reason for not using the custom `mv`.  In our custom
# code, backpropagation procedure is determined automatically, while in
# `torch.mv` it is implemented manually for the sake of efficiency and
# numerical precision.  We will get back to the latter topic in a couple of
# sessions.
##############################################################################


# Acknowledgement: parts of the code borrowed from one of your solutions
def benchmark(with_gradient: bool = False):
    """Set `with_gradient` to True if you want to benchmark backpropagation."""
    vector = torch.randn(25,dtype=torch.float, requires_grad=True)
    matrix = torch.randn(5000,25,dtype=torch.float, requires_grad=True)

    # TODO: to do proper benchmarking we should probably repeat the
    # experiment below a number of times

    start_time = timeit.default_timer()
    y = mv(matrix, vector)
    if with_gradient:
        y.sum().backward()
        print(vector.grad)
    print("mv:", timeit.default_timer()-start_time)

    if with_gradient:
        vector.grad.zero_()
        matrix.grad.zero_()

    start_time = timeit.default_timer()
    y = torch.mv(matrix, vector)
    if with_gradient:
        y.sum().backward()
        print(vector.grad)
    print("torch.mv:", timeit.default_timer()-start_time)
