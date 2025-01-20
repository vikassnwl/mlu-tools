import os
import cv2
from mlu_tools.utils import extract_num_from_end
import shutil
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import math
from mlu_tools.utils import get_dynamic_path, count_files
from mlu_tools.validation import (
    validate_pixel_range_0_255,
    validate_array_like,
    validate_dir,
)


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


class TFDataGenerator:
    def __init__(self, **kwargs):
        valid_args = {
            "brightness_range",
            "contrast_range",
            "horizontal_flip",
            "rotation_range",
            "height_shift_range",
            "width_shift_range",
            "zoom_range",
            "hue_range",
            "saturation_range",
            "rescale",
        }
        invalid_args = set(kwargs) - valid_args
        if invalid_args:
            raise Exception(f"Invalid arguments: {invalid_args}")
        self.kwargs = kwargs
        fill_mode = kwargs.get("fill_mode", "reflect")

        self.transformations = []
        if "brightness_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.RandomBrightness(kwargs["brightness_range"])
            )  # valid factor 0-1
        if "contrast_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.RandomContrast(kwargs["contrast_range"])
            )  # [1-lower, 1+upper], lower=upper if factor is a single value
        if "horizontal_flip" in kwargs and kwargs["horizontal_flip"] == True:
            self.transformations.append(tf.keras.layers.RandomFlip("horizontal"))
        if "rotation_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.RandomRotation(
                    kwargs["rotation_range"] / 360, fill_mode=fill_mode
                )
            )
        if "height_shift_range" in kwargs or "width_shift_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.RandomTranslation(
                    kwargs.get("height_shift_range", 0),
                    kwargs.get("width_shift_range", 0),
                    fill_mode=fill_mode,
                )
            )
        if "zoom_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.RandomZoom(kwargs["zoom_range"], fill_mode=fill_mode)
            )
        if "hue_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.Lambda(
                    lambda X: tf.image.random_hue(X, kwargs["hue_range"])
                )
            )
        if "saturation_range" in kwargs:
            self.transformations.append(
                tf.keras.layers.Lambda(
                    lambda X: tf.image.random_saturation(X, *kwargs["saturation_range"])
                )
            )

        # Rescale in the end of all the transformations
        if "rescale" in kwargs:
            self.transformations.append(tf.keras.layers.Rescaling(kwargs["rescale"]))

    def data_loader(
        self,
        X,
        y=None,
        buffer_size=1000,
        batch_size=32,
        expansion_factor=None,
        **kwargs,
    ):
        if isinstance(X, str):
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                X, batch_size=batch_size, **kwargs
            )
        else:
            if "brightness_range" in self.kwargs or "contrast_range" in self.kwargs:
                custom_message = (
                    "In order to apply brightness or contrast adjustments to the input image, pixel values must be in the range of 0 to 255."
                    "\nEnsure that the image has appropriate pixel values, and if needed, rescale it using the rescale argument of TFDataGenerator."
                )
                validate_pixel_range_0_255(X, custom_message=custom_message)
            # Create a tf.data.Dataset from the variables
            if y is None:
                dataset = tf.data.Dataset.from_tensor_slices(X)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((X, y))
            # Shuffle and batch the dataset
            dataset = dataset.shuffle(buffer_size).batch(batch_size)
        if self.transformations != []:
            # Apply data augmentation to the dataset
            transformations_pipeline = tf.keras.Sequential(self.transformations)

            def wrapper1(*args):
                return transformations_pipeline(args[0])

            def wrapper2(*args):
                return transformations_pipeline(args[0]), args[1]

            wrapper = wrapper1 if y is None and not isinstance(X, str) else wrapper2
            dataset = dataset.map(
                wrapper,
                num_parallel_calls=tf.data.AUTOTUNE,  # Enable parallel processing
            )
            if expansion_factor:
                # Create an additional dataset that contains 50% of the original dataset
                total_items = count_files(X) if isinstance(X, str) else len(X)
                additional_data = dataset.take(
                    int(np.ceil(total_items * expansion_factor / batch_size))
                ).map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.concatenate(additional_data)

        # Add prefetching for performance
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def flow(self, X, y=None, buffer_size=1000, batch_size=32, expansion_factor=None):
        validate_array_like(X)
        return self.data_loader(
            X, y, buffer_size, batch_size, expansion_factor=expansion_factor
        )

    def flow_from_directory(self, X, batch_size=32, expansion_factor=None, **kwargs):
        validate_dir(X)
        return self.data_loader(
            X, batch_size=batch_size, expansion_factor=expansion_factor, **kwargs
        )


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
