# Utility script for video processing for human detection
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add the project root directory to the Python path
# This is to ensure that 'from models.human_detector import HumanDetector' works
# when running this script directly or when imported by other scripts (e.g. in Streamlit app)
# if Streamlit app is run from project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # This assumes scripts/ is one level down from project root
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from models.human_detector import HumanDetector
except ImportError as e:
    print(f"Error: Could not import HumanDetector from models.human_detector: {e}")
    print(f"Current sys.path: {sys.path}")
    # Provide a dummy class if import fails, to allow basic script structure check
    class HumanDetector:
        def __init__(self): print("Dummy HumanDetector initialized due to import error.")
        def detect_humans(self, pil_image, threshold=0.5):
            print(f"Dummy detect_humans called for image, threshold {threshold}.")
            return []
    # sys.exit(1) # Optionally exit if the real detector is critical


def create_dummy_video(video_path, duration_sec=2, fps=30, width=640, height=480):
    """Creates a simple dummy MP4 video with a moving rectangle."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for path: {video_path}")
        return False

    num_frames = duration_sec * fps
    rect_size = 50
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background
        start_x = int((i * (width - rect_size)) / num_frames)
        start_y = int((i * (height - rect_size)) / num_frames)
        end_x = start_x + rect_size
        end_y = start_y + rect_size
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), -1) # Red rectangle
        out.write(frame)

    out.release()
    print(f"Dummy video '{video_path}' created successfully.")
    return True

def process_video_for_humans(detector: HumanDetector, input_video_path: str, output_video_path: str, detection_threshold: float = 0.5):
    """
    Processes a video to detect humans and draw bounding boxes and frame numbers.

    Args:
        detector: An instance of the HumanDetector class.
        input_video_path: Path to the input video file.
        output_video_path: Path to save the processed video.
        detection_threshold: Confidence threshold for human detection.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        print(f"Warning: Could not get valid FPS. Defaulting to 30 FPS.")
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output: {output_video_path}")
        cap.release()
        return

    print(f"Processing video: {input_video_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR frame to RGB, then to PIL Image for detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Call detect_humans on the PIL Image using the provided detector instance
        detections = detector.detect_humans(pil_image, threshold=detection_threshold)

        # Iterate through returned detections (box, score)
        for box, score in detections: # Each detection is (box, score)
            xmin, ymin, xmax, ymax = [int(coord) for coord in box]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Green box
            # Optionally, put score text
            # score_text = f"{score:.2f}"
            # cv2.putText(frame, score_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


        # Add Frame Number
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # White text

        out.write(frame)

        if frame_count % (fps * 5) == 0: # Log every 5 seconds of video processed
            print(f"Processed {frame_count} frames ({frame_count // fps} seconds of video)...")

    cap.release()
    out.release()
    print(f"Finished processing. Output video saved to: {output_video_path}")


if __name__ == '__main__':
    test_input_video_path = 'test_dummy_video.mp4'
    test_output_video_path = 'output_dummy_video_with_frames.mp4'

    print("Instantiating HumanDetector for the main test block...")
    # This might take time if model weights need to be downloaded
    try:
        human_detector_instance = HumanDetector()
    except Exception as e:
        print(f"Failed to instantiate HumanDetector: {e}")
        print("Exiting script as detector is crucial for processing.")
        sys.exit(1)

    print("Attempting to create a dummy video for testing...")
    if not create_dummy_video(test_input_video_path, duration_sec=2, fps=30):
        print("Failed to create dummy video. Exiting.")
        sys.exit(1)

    if not os.path.exists(test_input_video_path):
        print(f"Dummy video not found at {test_input_video_path}. Exiting.")
        sys.exit(1)
    else:
        print(f"Dummy video found at {test_input_video_path}.")

    print(f"Starting video processing for {test_input_video_path}...")
    try:
        process_video_for_humans(human_detector_instance, test_input_video_path, test_output_video_path)
        print(f"Video processing complete. Check '{test_output_video_path}'.")
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy video
        # if os.path.exists(test_input_video_path):
        #     os.remove(test_input_video_path)
        # if os.path.exists(test_output_video_path): # Or keep it for inspection
        #     os.remove(test_output_video_path)
        pass
    print("Main script execution finished.")
