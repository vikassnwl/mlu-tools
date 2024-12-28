import tensorflow as tf


def handle_symbolic_tensor(out_dtypes=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal out_dtypes
            out_dtypes = out_dtypes or list(map(lambda x: x.dtype, args))
            return tf.numpy_function(func, [*args], out_dtypes)
        return wrapper
    return decorator