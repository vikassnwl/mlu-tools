import os
import cv2
import numpy as np


def classify_video(input_video_path,
                   preprocess_frame, 
                   model, 
                   labels_dict,
                   default_output_dir="outputs",
                   output_filename=None, 
                   save=False, 
                   show=True
                   ):
    """
    Process a video by writing text on each frame and saving the result as a new video.

    Args:
        input_video_path (str): Path to the input video file.
        preprocess_frame (func): Function to preprocess each frame
        model (keras model): A classification model
        labels_dict (list, dict): A list or dictionary having class names
        default_output_dir (str): An output directory where video is to be saved
        output_filename (str): Output video file's name
        save (bool): True for saving the video otherwise False
        show (bool): True for showing each processed frame otherwise False

    Returns:
        list(tuple): A list of tuples containing prediction info for each frame.
                     (probas, pred_label, pred_class, pred_conf)
    """

    os.makedirs(default_output_dir, exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    if save:
        if not output_filename:
            basename = os.path.basename(input_video_path)
            basename_wo_ext, ext = os.path.splitext(basename)
            vid_pth_wo_ext = f"{default_output_dir}/{basename_wo_ext}"
            save_as = f"{vid_pth_wo_ext}_classified{ext}"
        else:
            save_as = f"{default_output_dir}/{output_filename}"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

        # Create VideoWriter object
        out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)

    window_w = 720
    dynamic_scale = int(width/window_w)

    if show:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Allow resizing
        window_h = int(window_w*height/width)
        cv2.resizeWindow("Frame", window_w, window_h)  # Set the window size


    prediction_info = []
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # make prediction
        frame_mod = preprocess_frame(frame)
        probas = model.predict(np.expand_dims(frame_mod, 0), verbose=0)[0]
        pred_label = probas.argmax()
        pred_class = labels_dict[pred_label]
        pred_conf = probas[pred_label]

        prediction_info.append((probas, pred_label, pred_class, pred_conf))

        # Write text on the frame
        position = (50, 50*dynamic_scale)  # Text position (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1*dynamic_scale
        color = (0, 255, 0)  # Green text
        thickness = 2*dynamic_scale

        cv2.putText(frame, pred_class, position, font, font_scale, color, thickness)
        cv2.putText(frame, f"{pred_conf:.2f}", (50, 100*dynamic_scale), font, font_scale, color, thickness)

        # Display the frame
        if show:
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):  # Press 'q' to exit
            break

        # Write the frame to the output video
        if save: out.write(frame)

    # Release resources
    cap.release()
    if save:
        print(f"Processed video saved at {save_as}")
        out.release()
    if show: cv2.destroyAllWindows()

    return prediction_info