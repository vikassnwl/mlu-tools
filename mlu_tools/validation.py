import os
from collections.abc import Iterable


def validate_file(path, arg_name, valid_types=[]):
    if not os.path.exists(path):
        raise Exception(f"{arg_name}='{path}' does not exist!")
    
    if not os.path.isfile(path):
        raise Exception(f"{arg_name}='{path}' is not a file!")

    if not valid_types: return

    for valid_type in valid_types:
        path_wo_ext, ext = os.path.splitext(path)
        ext = ext.lower()
        if valid_type == "vid":
            video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg'}
            if ext in video_extensions: break
        elif valid_type == "img":
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
            if ext in image_extensions: break
        elif valid_type == "txt":
            if ext == ".txt": break
        elif valid_type == "csv":
            if ext == ".csv": break
        else:
            raise Exception(f"type='{valid_type}' is not implemented for function `{validate_file.__name__}`")
    else:
        raise Exception(f"{arg_name}='{path}' is not in the specified valid_types:\n{valid_types}")
    
    return valid_type
    

def validate_dir(path, arg_name):
    if not os.path.exists(path):
        raise Exception(f"{arg_name}='{path}' does not exist!")
    if not os.path.isdir(path):
        raise Exception(f"{arg_name}='{path}' is not a directory!")
    

def validate_array_like(obj, arg_name, custom_message=""):
    custom_message = custom_message or f"'{arg_name}' must be an array-like object i.e., list, tuple, np.ndarray, etc., not {type(obj).__name__}."
    if not isinstance(obj, Iterable) or isinstance(obj, str):
        raise Exception(custom_message)