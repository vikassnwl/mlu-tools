import random
import tensorflow as tf
import numpy as np
import requests
from tqdm import tqdm


def set_global_seed(seed_value):
    # Set random seed for Python's random module
    random.seed(seed_value)
    
    # Set random seed for NumPy
    np.random.seed(seed_value)
    
    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)


def download(download_url, file_save_path):
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(file_save_path, "wb") as file:
        with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))