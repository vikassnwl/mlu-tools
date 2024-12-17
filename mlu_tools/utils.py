import random
import tensorflow as tf
import numpy as np
import requests
from tqdm import tqdm
import os
from .data_structure import TreeNode


def set_global_seed(seed_value):
    # Set random seed for Python's random module
    random.seed(seed_value)
    
    # Set random seed for NumPy
    np.random.seed(seed_value)
    
    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)


def download(file_url, file_save_path):
    FILE_ID = file_url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(file_save_path, "wb") as file:
        with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))


def tree(dir_pth):
    nodes = {}

    for dirpath, dirnames, filenames in sorted(os.walk(dir_pth)):
        if os.path.dirname(dirpath) not in nodes:
            nodes[dirpath] = TreeNode(os.path.basename(dirpath))
        else:
            if len(filenames):
                child_node = TreeNode(f"{os.path.basename(dirpath)} - {len(filenames)}")
            else:
                child_node = TreeNode(f"{os.path.basename(dirpath)}")
            nodes[os.path.dirname(dirpath)].add_child(child_node)
            nodes[dirpath] = child_node

    nodes[dir_pth].display()