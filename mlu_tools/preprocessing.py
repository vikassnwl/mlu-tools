import tensorflow as tf


def random_rotate(image, rotation_range):
    """Rotate the image within the specified range and fill blank space with nearest neighbor."""
    # Generate a random rotation angle in radians
    theta = tf.random.uniform([], -rotation_range, rotation_range) * tf.constant(3.14159265 / 180, dtype=tf.float32)

    # Get the image dimensions
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Create the rotation matrix
    rotation_matrix = tf.stack([
        [tf.cos(theta), -tf.sin(theta), 0],
        [tf.sin(theta),  tf.cos(theta), 0],
        [0, 0, 1]
    ])

    # Adjust for center-based rotation
    translation_to_origin = tf.stack([
        [1, 0, -width / 2],
        [0, 1, -height / 2],
        [0, 0, 1]
    ])
    
    translation_back = tf.stack([
        [1, 0, width / 2],
        [0, 1, height / 2],
        [0, 0, 1]
    ])

    # Cast matrices to tf.float32 for compatibility
    rotation_matrix = tf.cast(rotation_matrix, tf.float32)
    translation_to_origin = tf.cast(translation_to_origin, tf.float32)
    translation_back = tf.cast(translation_back, tf.float32)

    # Perform matrix multiplication
    transform_matrix = tf.linalg.matmul(translation_back, tf.linalg.matmul(rotation_matrix, translation_to_origin))

    # Extract the affine part of the transformation matrix (2x3 matrix for 2D transformation)
    affine_matrix = transform_matrix[:2, :]

    # Flatten the matrix into a 1D array and add [0, 0] to make it 8 elements
    affine_matrix_8 = tf.concat([affine_matrix[0, :], affine_matrix[1, :], [0, 0]], axis=0)

    # Apply the transformation with `fill_mode="nearest"`
    rotated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(affine_matrix_8, [1, 8]),
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST"
    )

    return tf.squeeze(rotated_image)


def random_translate(image, width_factor, height_factor):
    """Randomly translate the image horizontally and vertically within the specified factors.
    
    Args:
        image: Input image tensor.
        width_factor: Horizontal shift factor (0.1 means 10% of width).
        height_factor: Vertical shift factor (0.1 means 10% of height).
    
    Returns:
        Translated image.
    """
    # Get the image dimensions
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Convert factors to tensors and cast them to float32
    width_factor = tf.cast(width_factor, tf.float32)
    height_factor = tf.cast(height_factor, tf.float32)

    # Cast image dimensions to float32 to match the factor types
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # Calculate the maximum shifts based on the image dimensions
    max_width_shift = width * width_factor
    max_height_shift = height * height_factor

    # Generate random translation values within the given factors
    tx = tf.random.uniform([], -max_width_shift, max_width_shift, dtype=tf.float32)
    ty = tf.random.uniform([], -max_height_shift, max_height_shift, dtype=tf.float32)

    # Create the translation matrix as a 1D array with 8 values
    # [a, b, tx, d, e, ty, 0, 0]
    translation_matrix = tf.concat([
        tf.ones([1], dtype=tf.float32),  # a = 1
        tf.zeros([1], dtype=tf.float32),  # b = 0
        [tx],                             # tx (horizontal shift)
        tf.zeros([1], dtype=tf.float32),  # d = 0
        tf.ones([1], dtype=tf.float32),   # e = 1
        [ty],                             # ty (vertical shift)
        tf.zeros([2], dtype=tf.float32)   # [0, 0]
    ], axis=0)

    # Apply the translation with `fill_mode="nearest"`
    translated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(translation_matrix, [1, 8]),  # Ensure 8 values
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST"
    )

    return tf.squeeze(translated_image)