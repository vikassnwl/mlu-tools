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
import cv2
from IPython.display import FileLink
from datetime import datetime
from mega import Mega


def set_global_seed(seed_value):
    # Set random seed for Python's random module
    random.seed(seed_value)

    # Set random seed for NumPy
    np.random.seed(seed_value)

    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)


def download(file_url, file_save_path, download_from="drive", force=False):
    dest_path = os.path.dirname(file_save_path)
    dest_filename = os.path.basename(file_save_path)
    print(f"Downloading {dest_filename} to {dest_path}/...")
    if os.path.exists(file_save_path) and not force:
        # print("File already exists!\n")
        print(f"{file_save_path} already exists. Skipping download.\n")
        return
    
    if dest_path:
        os.makedirs(dest_path, exist_ok=True)

    if download_from == "web":
        response = requests.get(file_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(file_save_path, "wb") as file:
            with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    bar.update(len(chunk))
    elif download_from == "mega":
        # Initialize Mega object
        mega = Mega()
        # Download the file
        file = mega.download_url(file_url, dest_path=(dest_path or "."), 
                        dest_filename=dest_filename)
        print(f"File downloaded at {file}\n")
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


def download_yt_playlist(playlist_url, quality="best"):
    # Configure download options
    format_mapping = {
        "low": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]",
        "medium": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]",
        "high": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]",
        "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
    }

    options = {
        "outtmpl": os.path.join("downloads", "%(title)s.%(ext)s"),
        "format": format_mapping.get(quality, "best"),
        "ignoreerrors": True,  # Ignore errors like private videos
        "no_post_overwrites": True,  # Avoid overwriting the existing files
    }

    # Download the playlist
    with YoutubeDL(options) as ydl:
        ydl.download([playlist_url])

    print(f"Playlist downloaded successfully as {quality} quality MP4!")


def unpack_archive(file_path, target_dir=None, force=False):
    target_dir = target_dir or (os.path.dirname(file_path) if os.path.dirname(file_path) else ".")
    
    # Determine archive type and get root dir
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            members = zip_ref.namelist()

    elif file_path.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(file_path, "r") as tar_ref:
            members = tar_ref.getnames()

    else:
        print("Unsupported file type")
        return

    # CHECKING IF ARCHIVE CONTAINS A ROOT FOLDER OR NOT
    archive_contains_a_root_folder = all([os.path.dirname(member) for member in members])
    if archive_contains_a_root_folder:
        root_dir = os.path.dirname(members[0])
    else:
        root_dir = os.path.splitext(os.path.basename(file_path))[0]

    unpacked_dir = os.path.join(target_dir, root_dir)
    print(f"Unpacking {file_path} to {unpacked_dir}...")
    if os.path.exists(unpacked_dir) and not force:
        print(f"{unpacked_dir} already exists. Skipping unpacking.\n")
        return

    if not archive_contains_a_root_folder:
        os.makedirs(unpacked_dir)

    # Unpack with progress bar
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for member in tqdm(members, desc="Extracting", unit="file"):
                if archive_contains_a_root_folder:
                    zip_ref.extract(member, target_dir)
                else:
                    zip_ref.extract(member, unpacked_dir)
    elif file_path.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(file_path, "r") as tar_ref:
            for member in tqdm(members, desc="Extracting", unit="file"):
                if archive_contains_a_root_folder:
                    tar_ref.extract(member, target_dir)
                else:
                    tar_ref.extract(member, unpacked_dir)

    print(f"Archive unpacked to {unpacked_dir}\n")
    return unpacked_dir


def count_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files


def video_capture(src, frame_processing_func=None):
    # Open the video file
    cap = cv2.VideoCapture(src)

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the delay in seconds between frames
    frame_delay = 1 / fps

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_processing_func:
            # Process the frame (e.g., convert to grayscale)
            frame = frame_processing_func(frame)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Wait for the appropriate time based on the FPS
        key = cv2.waitKey(int(frame_delay * 1000))  # Convert seconds to milliseconds
        if key == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def create_download_link(file_path):
    """Create a download link for the given file path."""
    return FileLink(file_path)


def extract_num_from_end(filename):
    filename_wo_ext = os.path.splitext(filename)[0]
    extracted_num = ""
    for c in filename_wo_ext[::-1]:
        try:
            int(c)
            extracted_num += c
        except:
            break
    
    return int(extracted_num[::-1])


def get_datetime_str():
    # Get the current datetime
    current_time = datetime.now()
    # Format the datetime string
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
    return datetime_str
