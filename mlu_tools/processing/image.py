import os
import cv2
from mlu_tools.utils import extract_num_from_end, get_dynamic_path
import shutil
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import math


def frames2vid(frames_dir, output_video_path, fps):
    frame = cv2.imread(f"{frames_dir}/{os.listdir(frames_dir)[0]}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    frame_size = frame.shape[1::-1]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    for filename in sorted(os.listdir(frames_dir), key=extract_num_from_end):
        file_path = f"{frames_dir}/{filename}"
        frame = cv2.imread(file_path)
        out.write(frame)
    out.release()


def perform_undersampling(dir_pth):
    shutil.copytree(dir_pth, f"{dir_pth}_undersampled")
    dir_pth = f"{dir_pth}_undersampled"
    min_files_cnt = math.inf
    for subdir in os.listdir(dir_pth):
        subdir_pth = f"{dir_pth}/{subdir}"
        min_files_cnt = min(min_files_cnt, len(os.listdir(subdir_pth)))

    for subdir in os.listdir(dir_pth):
        subdir_pth = f"{dir_pth}/{subdir}"
        filenames = os.listdir(subdir_pth)
        remove_filenames = random.sample(
            os.listdir(subdir_pth), len(filenames) - min_files_cnt
        )
        for filename in remove_filenames:
            filepath = f"{subdir_pth}/{filename}"
            os.remove(filepath)


def perform_oversampling(dir_pth, target_size):
    if not os.path.exists(f"{dir_pth}_oversampled"):
        shutil.copytree(dir_pth, f"{dir_pth}_oversampled")
    dir_pth = f"{dir_pth}_oversampled"
    max_files_cnt = -math.inf
    for subdir in os.listdir(dir_pth):
        subdir_pth = f"{dir_pth}/{subdir}"
        max_files_cnt = max(max_files_cnt, len(os.listdir(subdir_pth)))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,  # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of width
        height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of height
        shear_range=20,  # Apply shear transformations with a shear angle of 20 degrees
        zoom_range=0.2,  # Randomly zoom images in/out by up to 20%
        horizontal_flip=True,  # Randomly flip images horizontally
    )

    for subdir in os.listdir(dir_pth):
        subdir_pth = f"{dir_pth}/{subdir}"
        filenames = os.listdir(subdir_pth)
        file_paths = [os.path.join(subdir_pth, img) for img in filenames]
        df = pd.DataFrame({"filename": file_paths})
        num_img_in_dir = len(filenames)
        num_img_to_gen = max_files_cnt - num_img_in_dir
        image_generator = datagen.flow_from_dataframe(
            dataframe=df,
            x_col="filename",
            y_col=None,
            class_mode=None,
            batch_size=num_img_to_gen,
            target_size=target_size,
        )
        num_iters = int(np.ceil(num_img_to_gen / num_img_in_dir))
        while num_iters:
            images = next(image_generator)

            # for images in image_generator: break
            for image in images:
                filename = "oversampled.jpg"
                filepath = f"{subdir_pth}/{filename}"
                cv2.imwrite(get_dynamic_path(filepath), image[..., ::-1])

            num_iters -= 1


def process_path(file_path, image_size, crop_to_aspect_ratio):
    label = tf.strings.split(file_path, os.sep)[-2]
    label = int(label)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, image_size)  # doesn't support center cropping
    img = tf.keras.layers.Resizing(
        *image_size, crop_to_aspect_ratio=crop_to_aspect_ratio
    )(img)

    return img, label


def get_affine_transform_vector(
    theta, tx, ty, shear, zx, zy, image_height, image_width
):
    # theta = math.radians(theta)
    theta = tf.math.multiply(theta, tf.constant(math.pi / 180.0))
    # shear = math.radians(shear)
    shear = tf.math.multiply(shear, tf.constant(math.pi / 180.0))

    # Center of the image
    cx, cy = image_width / 2.0, image_height / 2.0

    # Rotation
    rotation = tf.convert_to_tensor(
        [[tf.cos(theta), -tf.sin(theta)], [tf.sin(theta), tf.cos(theta)]],
        dtype=tf.float32,
    )

    # Shear
    shear_matrix = tf.convert_to_tensor(
        [[1.0, -tf.sin(shear)], [0.0, tf.cos(shear)]], dtype=tf.float32
    )

    # Zoom
    zoom = tf.convert_to_tensor([[zx, 0.0], [0.0, zy]], dtype=tf.float32)

    # Composite transform
    transform = tf.linalg.matmul(rotation, tf.linalg.matmul(shear_matrix, zoom))

    # Offset for center + translation
    offset = tf.convert_to_tensor(
        [
            cx - (transform[0, 0] * cx + transform[0, 1] * cy) + tx,
            cy - (transform[1, 0] * cx + transform[1, 1] * cy) + ty,
        ],
        dtype=tf.float32,
    )

    # 8-element vector with 2 zeros at the end
    flat_transform = tf.stack(
        [
            transform[0, 0],
            transform[0, 1],
            offset[0],
            transform[1, 0],
            transform[1, 1],
            offset[1],
            0.0,
            0.0,
        ]
    )

    return flat_transform


def apply_affine_transform_tf(
    x,
    rotation_range=0,
    height_shift_range=0,
    width_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode="NEAREST",
    interpolation="BILINEAR",
    fill_value=0.0,
):
    """
    Apply affine transform to a single image tensor (H, W, C).
    """

    # print(x.shape)

    if rotation_range:
        # theta = np.random.uniform(-rotation_range, rotation_range)
        theta = tf.random.uniform([], minval=-rotation_range, maxval=rotation_range)
    else:
        theta = 0.0

    if width_shift_range:
        ty = tf.random.uniform([], -width_shift_range, width_shift_range)
        ty *= tf.cast(tf.shape(x)[1], tf.float32)
    else:
        ty = tf.constant(0.0, dtype=tf.float32)

    if height_shift_range:
        tx = tf.random.uniform([], -height_shift_range, height_shift_range)
        tx *= tf.cast(tf.shape(x)[0], tf.float32)
    else:
        tx = tf.constant(0.0, dtype=tf.float32)

    if shear_range:
        shear = tf.random.uniform([], -shear_range, shear_range)
    else:
        shear = 0.0

    zoom_range = [1 - zoom_range, 1 + zoom_range]
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        ## Non-Uniform Zooming across Width and Height
        zx, zy = tf.unstack(tf.random.uniform([2], zoom_range[0], zoom_range[1]))
        ## Uniform Zooming across Width and Height
        # zx = tf.random.uniform([], zoom_range[0], zoom_range[1])
        # zy = zx

    if horizontal_flip and tf.random.uniform([], 0.0, 1.0) < 0.5:
        x = tf.reverse(x, axis=[1])

    if len(x.shape) != 3:
        raise ValueError("Input must be a 3D tensor [height, width, channels]")

    shape = tf.shape(x)
    height, width = shape[0], shape[1]

    transform = get_affine_transform_vector(
        theta,
        tx,
        ty,
        shear,
        zx,
        zy,
        tf.cast(height, tf.float32),
        tf.cast(width, tf.float32),
    )

    transform = tf.reshape(transform, [1, 8])  # shape: [1, 8]
    x = tf.expand_dims(x, axis=0)  # shape: [1, H, W, C]

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=x,
        transforms=transform,
        output_shape=tf.stack([height, width]),
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )

    return tf.squeeze(transformed, axis=0)  # shape: [H, W, C]


class TFDataGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def flow_from_directory(
        self, dir_pth, shuffle=True, image_size=(256, 256), crop_to_aspect_ratio=False
    ):
        dataset = tf.data.Dataset.list_files(f"{dir_pth}/*/*", shuffle=shuffle)
        dataset = dataset.map(
            lambda x: process_path(x, image_size, crop_to_aspect_ratio),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if self.kwargs:
            dataset = dataset.map(
                lambda x, y: (apply_affine_transform_tf(x, **self.kwargs), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return dataset


def random_rotate(image, rotation_range):
    """Rotate the image within the specified range and fill blank space with nearest neighbor."""
    # Generate a random rotation angle in radians
    theta = tf.random.uniform([], -rotation_range, rotation_range) * tf.constant(
        3.14159265 / 180, dtype=tf.float32
    )

    # Get the image dimensions
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Create the rotation matrix
    rotation_matrix = tf.stack(
        [
            [tf.cos(theta), -tf.sin(theta), 0],
            [tf.sin(theta), tf.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Adjust for center-based rotation
    translation_to_origin = tf.stack(
        [[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]
    )

    translation_back = tf.stack([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]])

    # Cast matrices to tf.float32 for compatibility
    rotation_matrix = tf.cast(rotation_matrix, tf.float32)
    translation_to_origin = tf.cast(translation_to_origin, tf.float32)
    translation_back = tf.cast(translation_back, tf.float32)

    # Perform matrix multiplication
    transform_matrix = tf.linalg.matmul(
        translation_back, tf.linalg.matmul(rotation_matrix, translation_to_origin)
    )

    # Extract the affine part of the transformation matrix (2x3 matrix for 2D transformation)
    affine_matrix = transform_matrix[:2, :]

    # Flatten the matrix into a 1D array and add [0, 0] to make it 8 elements
    affine_matrix_8 = tf.concat(
        [affine_matrix[0, :], affine_matrix[1, :], [0, 0]], axis=0
    )

    # Apply the transformation with `fill_mode="nearest"`
    rotated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(affine_matrix_8, [1, 8]),
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
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
    translation_matrix = tf.concat(
        [
            tf.ones([1], dtype=tf.float32),  # a = 1
            tf.zeros([1], dtype=tf.float32),  # b = 0
            [tx],  # tx (horizontal shift)
            tf.zeros([1], dtype=tf.float32),  # d = 0
            tf.ones([1], dtype=tf.float32),  # e = 1
            [ty],  # ty (vertical shift)
            tf.zeros([2], dtype=tf.float32),  # [0, 0]
        ],
        axis=0,
    )

    # Apply the translation with `fill_mode="nearest"`
    translated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(translation_matrix, [1, 8]),  # Ensure 8 values
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
    )

    return tf.squeeze(translated_image)
