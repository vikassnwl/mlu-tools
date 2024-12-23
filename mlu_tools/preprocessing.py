import tensorflow as tf
import os
import concurrent.futures
import cv2
import shutil
import math
import random
import numpy as np
import pandas as pd
from .utils import get_dynamic_path
from tqdm import tqdm


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


def vid2frames(inp_vid_pth, out_frames_dir):
    os.makedirs(out_frames_dir, exist_ok=True)
    inp_vid_name = os.path.basename(inp_vid_pth)
    inp_vid_name_wo_ext, ext = os.path.splitext(inp_vid_name)
    vidcap = cv2.VideoCapture(inp_vid_pth)
    is_success, image = vidcap.read()
    frame_number = 0

    while is_success:
        save_path_and_name = f"{out_frames_dir}/{inp_vid_name_wo_ext}_frame-{frame_number}.jpg"
        cv2.imwrite(save_path_and_name, image)
        is_success, image = vidcap.read()
        frame_number += 1


def vids2frames(vids_dir, frames_dir, execution_mode="multi-processing"):
    vid_pth_list = []
    frames_pth_list = []
    for dirpath, dirnames, filenames in os.walk(vids_dir):
        if len(filenames):
            frames_pth = f"{frames_dir}/{dirpath.split(vids_dir)[-1]}"
            for filename in filenames:
                filepth = f"{dirpath}/{filename}"
                vid_pth_list.append(filepth)
                frames_pth_list.append(frames_pth)

    if execution_mode == "loop":
        # loop
        for inp_vid_pth, out_frames_dir in tqdm(list(zip(vid_pth_list, frames_pth_list))):
            vid2frames(inp_vid_pth, out_frames_dir)

    elif execution_mode == "multi-threading":
        # multi threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(vid2frames, vid_pth_list, frames_pth_list)
            for _ in tqdm(results, total=len(vid_pth_list)): pass

    elif execution_mode == "multi-processing":
        # multi processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(vid2frames, vid_pth_list, frames_pth_list)
            for _ in tqdm(results, total=len(vid_pth_list)): pass


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
        remove_filenames = random.sample(os.listdir(subdir_pth), len(filenames)-min_files_cnt)
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
        rotation_range=20,           # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,       # Randomly shift images horizontally by up to 20% of width
        height_shift_range=0.2,      # Randomly shift images vertically by up to 20% of height
        shear_range=0.2,             # Apply shear transformations with a shear intensity of 20%
        zoom_range=0.2,              # Randomly zoom images in/out by up to 20%
        horizontal_flip=True,        # Randomly flip images horizontally
    )

    for subdir in os.listdir(dir_pth):
        subdir_pth = f"{dir_pth}/{subdir}"
        filenames = os.listdir(subdir_pth)
        file_paths = [os.path.join(subdir_pth, img) for img in filenames]
        df = pd.DataFrame({"filename": file_paths})
        num_img_in_dir = len(filenames)
        num_img_to_gen = max_files_cnt-num_img_in_dir
        image_generator = datagen.flow_from_dataframe(
            dataframe=df,
            x_col="filename",
            y_col=None,
            class_mode=None,
            batch_size=num_img_to_gen,
            target_size=target_size
        )
        num_iters = int(np.ceil(num_img_to_gen/num_img_in_dir))
        while num_iters:
            images = next(image_generator)

            # for images in image_generator: break
            for image in images:
                filename = "oversampled.jpg"
                filepath = f"{subdir_pth}/{filename}"
                cv2.imwrite(get_dynamic_path(filepath), image[..., ::-1])
                
            num_iters -= 1