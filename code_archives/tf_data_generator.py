import tensorflow as tf
from mlu_tools.utils import get_dynamic_path, count_files
from mlu_tools.validation import (
    validate_pixel_range_0_255,
    validate_array_like,
    validate_dir,
)
import numpy as np



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