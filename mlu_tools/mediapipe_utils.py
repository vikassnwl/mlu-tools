import cv2
import mediapipe as mp
from threading import Thread
from queue import Queue, Empty
import numpy as np
import tempfile
from mlu_tools.validation import validate_file


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_frame(input_queue, output_queue):
    hands = mp_hands.Hands()
    while True:
        frame = input_queue.get()
        if frame is None:  # Stop signal
            break

        # Convert frame to RGB and process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        output_queue.put((frame, results))  # Send back the processed frame and results

    hands.close()


def detect_hand_landmarks(source):
    delay = 1
    if type(source) == np.ndarray:
        delay = 0
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
            cv2.imwrite(temp_file.name, source[..., ::-1])  # Save NumPy array as an image
            cap = cv2.VideoCapture(temp_file.name)
    else:
        if validate_file(source, valid_types=["img"], raise_exception=False):
            delay = 0
        cap = cv2.VideoCapture(source)
    input_queue = Queue(maxsize=1)  # Limit queue size to 1 for the latest frame
    output_queue = Queue(maxsize=1)  # Limit queue size to 1 for the latest results

    # Start the worker thread
    thread = Thread(target=process_frame, args=(input_queue, output_queue), daemon=True)
    thread.start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Try to enqueue the latest frame (discard older frames if queue is full)
        if not input_queue.full():
            input_queue.put(frame)

        # Render the latest processed frame
        try:
            processed_frame, results = output_queue.get(
                timeout=0.1
            )  # Use timeout to avoid blocking indefinitely
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            # Show the processed frame with hand landmarks
            cv2.imshow("Hand Tracking", processed_frame)
        except Empty:
            pass  # If no result is available, just continue

        # Check if 'q' is pressed for exiting
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        if delay == 0:
            # Render the last processed frame
            try:
                processed_frame, results = output_queue.get(
                    timeout=0.1
                )  # Use timeout to avoid blocking indefinitely
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                # Show the processed frame with hand landmarks
                cv2.imshow("Hand Tracking", processed_frame)
            except Empty:
                pass  # If no result is available, just continue

            cv2.waitKey(delay)


    # Clean up
    input_queue.put(None)  # Send stop signal to the worker thread
    thread.join()  # Wait for the worker thread to finish
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Ensure the window is properly closed
