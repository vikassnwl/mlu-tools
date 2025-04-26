import os
import cv2
from tqdm import tqdm
import concurrent.futures


def trim_video(
    vid_pth,
    default_output_dir="outputs",
    output_filename=None,
    trim_st_in_s=0,
    trim_et_in_s=5,
):
    os.makedirs(default_output_dir, exist_ok=True)
    if not output_filename:
        basename = os.path.basename(vid_pth)
        basename_wo_ext, ext = os.path.splitext(basename)
        vid_pth_wo_ext = f"{default_output_dir}/{basename_wo_ext}"
        save_as = f"{vid_pth_wo_ext}_trimmed_{trim_st_in_s}_{trim_et_in_s}{ext}"
    else:
        save_as = f"{default_output_dir}/{output_filename}"

    cap = cv2.VideoCapture(vid_pth)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)
    frames_to_process = (trim_et_in_s - trim_st_in_s) * fps
    while frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_to_process -= 1
    cap.release()
    out.release()
    return save_as


def vid2frames(inp_vid_pth, out_frames_dir):
    os.makedirs(out_frames_dir, exist_ok=True)
    inp_vid_name = os.path.basename(inp_vid_pth)
    inp_vid_name_wo_ext, ext = os.path.splitext(inp_vid_name)
    vidcap = cv2.VideoCapture(inp_vid_pth)
    is_success, image = vidcap.read()
    frame_number = 0

    while is_success:
        save_path_and_name = (
            f"{out_frames_dir}/{inp_vid_name_wo_ext}_frame-{frame_number}.jpg"
        )
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
        for inp_vid_pth, out_frames_dir in tqdm(
            list(zip(vid_pth_list, frames_pth_list))
        ):
            vid2frames(inp_vid_pth, out_frames_dir)

    elif execution_mode == "multi-threading":
        # multi threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(vid2frames, vid_pth_list, frames_pth_list)
            for _ in tqdm(results, total=len(vid_pth_list)):
                pass

    elif execution_mode == "multi-processing":
        # multi processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(vid2frames, vid_pth_list, frames_pth_list)
            for _ in tqdm(results, total=len(vid_pth_list)):
                pass
