import os
from collections.abc import Iterable
import inspect


# def get_param_name(value, caller_frame):
#     # Get the variable name from the caller's local scope
#     caller_locals = caller_frame.f_back.f_locals
#     param_name = [key for key, val in caller_locals.items() if val is value]
#     if param_name:
#         param_name = [get_param_name(value, caller_frame.f_back)]
#     else:
#         # Get the variable name from the callee's local scope
#         callee_locals = caller_frame.f_locals
#         param_name = [key for key, val in callee_locals.items() if val is value]
#     return param_name[0]

def get_param_name(value, caller_frame):
    # Get the variable name from the caller's local scope
    caller_locals = caller_frame.f_back.f_locals 
    param_name = [key for key, val in caller_locals.items() if val is value]
    return param_name[0]


def validate_file(path, valid_types=[]):
    param_name = get_param_name(path, inspect.currentframe())
    if not os.path.exists(path):
        raise Exception(f"{param_name}='{path}' does not exist!")
    
    if not os.path.isfile(path):
        raise Exception(f"{param_name}='{path}' is not a file!")

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
        raise Exception(f"{param_name}='{path}' is not in the specified valid_types:\n{valid_types}")
    
    return valid_type
    

def validate_dir(path):
    param_name = get_param_name(path, inspect.currentframe())
    if not isinstance(path, str):
        raise Exception(f"'{param_name}' must be a string not {type(path).__name__}.")
    if not os.path.exists(path):
        raise Exception(f"{param_name}='{path}' does not exist!")
    if not os.path.isdir(path):
        raise Exception(f"{param_name}='{path}' is not a directory!")
    

def validate_array_like(obj, custom_message=""):
    param_name = get_param_name(obj, inspect.currentframe())
    custom_message = custom_message or f"'{param_name}' must be an array-like object i.e., list, tuple, np.ndarray, etc., not {type(obj).__name__}."
    if not isinstance(obj, Iterable) or isinstance(obj, str):
        raise Exception(custom_message)
    

def validate_pixel_range_0_255(X, custom_message=""):
    param_name = get_param_name(X, inspect.currentframe())
    if X.min() < 0 or X.max() <= 1:
        default_message = f"Pixel values are not in the range 0..255 for argument '{param_name}'."
        message = custom_message or default_message
        raise Exception(message)