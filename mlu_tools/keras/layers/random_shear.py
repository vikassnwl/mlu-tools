import tensorflow as tf
import numpy as np
import scipy


class RandomShear(tf.keras.layers.Layer):
    def __init__(self, shear_range, fill_mode, **kwargs):
        super(RandomShear, self).__init__(**kwargs)
        self.shear_range = shear_range
        self.fill_mode = fill_mode

    def call(self, inputs, training=True):
        if training:  # Only apply shear during training
            return tf.numpy_function(self._apply_random_shear, [inputs], tf.float32)
        return inputs  # Pass-through during inference

    def compute_output_shape(self, input_shape):
        return input_shape

    def _apply_random_shear(self, X_in, row_axis=0, col_axis=1, channel_axis=2, order=1, cval=0.0):
        X_in_ = [X_in] if X_in.ndim==3 else X_in
        X_out = []
        shears = np.random.uniform(-self.shear_range, self.shear_range, len(X_in))  # Randomize shear each time
        for x, shear in zip(X_in_, shears):
            # Input sanity checks:
            # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
            # 2. channels must be either first or last dimension
            if np.unique([row_axis, col_axis, channel_axis]).size != 3:
                raise ValueError(
                    "'row_axis', 'col_axis', and 'channel_axis' must be distinct"
                )

            # shall we support negative indices?
            valid_indices = set([0, 1, 2])
            actual_indices = set([row_axis, col_axis, channel_axis])
            if actual_indices != valid_indices:
                raise ValueError(
                    f"Invalid axis' indices: {actual_indices - valid_indices}"
                )

            if x.ndim != 3:
                raise ValueError("Input arrays must be multi-channel 2D images.")
            if channel_axis not in [0, 2]:
                raise ValueError(
                    "Channels are allowed and the first and last dimensions."
                )

            transform_matrix = None

            if shear != 0:
                shear = np.deg2rad(shear)
                shear_matrix = np.array(
                    [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]]
                )
                if transform_matrix is None:
                    transform_matrix = shear_matrix
                else:
                    transform_matrix = np.dot(transform_matrix, shear_matrix)

            if transform_matrix is not None:
                h, w = x.shape[row_axis], x.shape[col_axis]
                transform_matrix = self._transform_matrix_offset_center(
                    transform_matrix, h, w
                )
                x = np.rollaxis(x, channel_axis, 0)

                # Matrix construction assumes that coordinates are x, y (in that order).
                # However, regular numpy arrays use y,x (aka i,j) indexing.
                # Possible solution is:
                #   1. Swap the x and y axes.
                #   2. Apply transform.
                #   3. Swap the x and y axes again to restore image-like data ordering.
                # Mathematically, it is equivalent to the following transformation:
                # M' = PMP, where P is the permutation matrix, M is the original
                # transformation matrix.
                if col_axis > row_axis:
                    transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
                    transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
                final_affine_matrix = transform_matrix[:2, :2]
                final_offset = transform_matrix[:2, 2]

                channel_images = [
                    scipy.ndimage.affine_transform(
                        x_channel,
                        final_affine_matrix,
                        final_offset,
                        order=order,
                        mode=self.fill_mode,
                        cval=cval,
                    )
                    for x_channel in x
                ]
                x = np.stack(channel_images, axis=0)
                x = np.rollaxis(x, 0, channel_axis + 1)
            # return x
            X_out.append(x)
        
        X_out = X_out[0] if X_in.ndim==3 else X_out
        return np.array(X_out, "float32")

    @staticmethod
    def _transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 - 0.5
        o_y = float(y) / 2 - 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix