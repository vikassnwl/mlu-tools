import mediapipe as mp
import cv2
import numpy as np
import os
from mlu_tools.validation import validate_file
from mlu_tools.utils import get_datetime_str


def _detect_hand_landmarks(image, flip=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        image = image[..., ::-1]
        image = np.ascontiguousarray(image)
        if flip: image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = image[..., ::-1]
        image = np.ascontiguousarray(image)
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),  # Circle style
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)  # Line style
                                        )

    return image, results


def detect_hand_landmarks(source, 
                          show_image=False, 
                          save=False, 
                          default_output_dir="outputs", 
                          output_filename=None,
                          camera_fps=13):
    if save: os.makedirs(default_output_dir, exist_ok=True)
    if type(source) == int or validate_file(source, ["vid"], raise_exception=False):
        # source is a camera index or path of a video file
        results = []
        flip = False
        if type(source) == int:
            flip = True
            output_filename = f"{get_datetime_str()}.mp4"

        cap = cv2.VideoCapture(source)

        if save:
            if not output_filename:
                basename = os.path.basename(source)
                basename_wo_ext, ext = os.path.splitext(basename)
                vid_pth_wo_ext = f"{default_output_dir}/{basename_wo_ext}"
                save_as = f"{vid_pth_wo_ext}_hand_landmarks.mp4"
            else:
                save_as = f"{default_output_dir}/{output_filename}"

            # Get video properties
            fps = camera_fps if camera_fps != "auto" else int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

            # Create VideoWriter object
            out = cv2.VideoWriter(save_as, fourcc, fps, frame_size)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_mod, result = _detect_hand_landmarks(frame, flip)
            results.append(result)

            cv2.imshow("Frame", frame_mod)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

            if save: out.write(frame_mod)

        cap.release()
        if save: out.release()
        cv2.destroyAllWindows()
        
    elif type(source) == np.ndarray or validate_file(source, ["img"], raise_exception=False):
        if type(source) == np.ndarray:
            # source is an image in the form of numpy array
            image = source
        else:
            # source is a path of an image file
            image = cv2.imread(source)

        image_mod, results = _detect_hand_landmarks(image)
        if show_image:
            cv2.imshow("Image", image_mod)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        if save:
            if type(source) == np.ndarray:
                output_filename = f"{get_datetime_str()}.mp4"
            if not output_filename:
                basename = os.path.basename(source)
                basename_wo_ext, ext = os.path.splitext(basename)
                img_pth_wo_ext = f"{default_output_dir}/{basename_wo_ext}"
                save_as = f"{img_pth_wo_ext}_hand_landmarks.jpg"
            else:
                save_as = f"{default_output_dir}/{output_filename}"
            cv2.imwrite(save_as, image_mod)
    
    return results
