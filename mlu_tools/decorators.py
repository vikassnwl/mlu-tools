import tensorflow as tf
from .validation import validate_array_like


def handle_symbolic_tensor(out_dtypes=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal out_dtypes
            out_dtypes = out_dtypes or list(map(lambda x: x.dtype, args))
            validate_array_like(out_dtypes, "out_dtypes")
            out_dtypes = out_dtypes[0] if len(out_dtypes)==1 else out_dtypes
            return tf.numpy_function(func, [*args], out_dtypes)
        return wrapper
    return decorator