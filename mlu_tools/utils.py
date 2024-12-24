import random
import tensorflow as tf
import numpy as np
import requests
from tqdm import tqdm
import os
from .data_structure import TreeNode
import gdown
from yt_dlp import YoutubeDL
import zipfile
import tarfile


def set_global_seed(seed_value):
    # Set random seed for Python's random module
    random.seed(seed_value)
    
    # Set random seed for NumPy
    np.random.seed(seed_value)
    
    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)


def download(file_url, file_save_path, download_from="drive", force=False):
    if os.path.exists(file_save_path) and not force:
        print("File already exists!")
        return
    if download_from == "web":
        response = requests.get(file_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(file_save_path, "wb") as file:
            with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    bar.update(len(chunk))
    else:
        FILE_ID = file_url.split("/")[-2]
        download_url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
        gdown.download(download_url, file_save_path)


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


def get_dynamic_path(path):
    path_wo_ext, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(path):
        path = f"{path_wo_ext}_{i}{ext}"
        i += 1
    
    return path


def download_yt_playlist(playlist_url):
    # Configure download options
    options = {
        'outtmpl': os.path.join('downloads', '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Download MP4 directly
        'ignoreerrors': True,  # Ignore errors like private videos
        'no_post_overwrites': True,  # Avoid overwriting the existing files
    }

    # Download the playlist
    with YoutubeDL(options) as ydl:
        ydl.download([playlist_url])

    print("Playlist downloaded successfully as MP4!")


def unpack_archive(file_path, target_dir=".", force=False):
    # Extract the root directory from the archive
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            root_dir = zip_ref.namelist()[0].split('/')[0]  # First part of the first file path
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r') as tar_ref:
            root_dir = tar_ref.getnames()[0].split('/')[0]  # First part of the first file path
    else:
        print("Unsupported file type")
        return
    
    # Check if the root directory already exists in the target directory
    unpacked_dir = os.path.join(target_dir, root_dir)
    if os.path.exists(unpacked_dir) and not force:
        print(f"{unpacked_dir} already exists. Skipping unpacking.")
        return

    # Unpack the archive
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(target_dir)

    print(f"Archive unpacked to {unpacked_dir}")
