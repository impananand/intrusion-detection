import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torch # To ensure torch environment is active, and for st.cache_resource to see it

# Ensure scripts and models can be found.
# This assumes streamlit is run from the project root.
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent #This should be /app, then .parent gives project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from scripts.process_video import process_video_for_humans
    from models.human_detector import HumanDetector
except ImportError as e:
    st.error(f"Failed to import necessary modules: {e}. Ensure scripts/models are in PYTHONPATH.")
    # Define placeholders if import fails, so app can still start for debugging
    def process_video_for_humans(detector, input_path, output_path, roi=None):
        st.error("process_video_for_humans is not available due to import error.")
        with open(output_path, 'w') as f: f.write("Simulated processed video.")
        return True
    class HumanDetector: # type: ignore
        def __init__(self): st.error("Dummy HumanDetector: Real one failed to import.")
        def detect_humans(self, img, threshold=0.5): return []


# Cache the HumanDetector instance
@st.cache_resource
def load_human_detector():
    """Loads and returns the HumanDetector instance."""
    st.write("Attempting to load Human Detector model (cached)...") # Will show only on first run/cache miss
    try:
        detector = HumanDetector()
        st.write("Human Detector model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading HumanDetector: {e}")
        # Fallback to a dummy detector if actual loading fails
        class DummyDetector:
            def detect_humans(self, img, threshold=0.5): return []
        detector = DummyDetector() # type: ignore
    return detector


# 1. Set page title
st.set_page_config(page_title="Intrusion Detection System")

def main():
    st.title("Intrusion Detection System")

    # Load the model using the cached function
    # This will only run HumanDetector.__init__() once per session or until cache is cleared.
    human_detector = load_human_detector()

    # Initialize session state variables
    if 'temp_video_path' not in st.session_state: st.session_state.temp_video_path = None
    if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
    if 'roi_coords' not in st.session_state: st.session_state.roi_coords = None
    if 'selected_frame_index' not in st.session_state: st.session_state.selected_frame_index = 0
    if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None

    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name: # New file uploaded
            # Clean up old temp input file
            if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                try: os.remove(st.session_state.temp_video_path)
                except: pass
            st.session_state.temp_video_path = None
            # Clean up old processed video file
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                try: os.remove(st.session_state.processed_video_path)
                except: pass
            st.session_state.processed_video_path = None
            st.session_state.roi_coords = None
            st.session_state.selected_frame_index = 0
            st.session_state.uploaded_file_name = uploaded_file.name

        st.write(f"Uploaded video: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {uploaded_file.size} bytes")

        if not st.session_state.temp_video_path or not os.path.exists(st.session_state.temp_video_path):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1] or '.mp4'
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.temp_video_path = tmp_file.name
            except Exception as e:
                st.error(f"Error creating temporary file: {e}"); return

        if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            st.subheader("Uploaded Video")
            st.video(st.session_state.temp_video_path)

            # ROI SELECTION (existing logic)
            cap = cv2.VideoCapture(st.session_state.temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                st.session_state.selected_frame_index = st.slider(
                    "Select frame for ROI", 0, total_frames - 1, st.session_state.selected_frame_index, key="frame_slider"
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.selected_frame_index)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    st.subheader("Draw ROI (Rectangle)")
                    canvas_width, canvas_height = pil_image.width, pil_image.height
                    if pil_image.width > 700: # Scale down if too wide
                        ratio = 700 / pil_image.width
                        canvas_width, canvas_height = 700, int(pil_image.height * ratio)

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="red",
                        background_image=pil_image, drawing_mode="rect",
                        width=canvas_width, height=canvas_height, key="roi_canvas"
                    )
                    if canvas_result.json_data and canvas_result.json_data["objects"]:
                        rect = canvas_result.json_data["objects"][-1]
                        scale_x, scale_y = pil_image.width/canvas_width, pil_image.height/canvas_height
                        st.session_state.roi_coords = [int(rect["left"]*scale_x), int(rect["top"]*scale_y),
                                                       int((rect["left"]+rect["width"])*scale_x), int((rect["top"]+rect["height"])*scale_y)]
                        st.write("ROI (xmin, ymin, xmax, ymax):", st.session_state.roi_coords)
                else: st.error("Could not read frame for ROI.")
                cap.release()
            else: st.warning("Video has no frames for ROI selection.")

            # PROCESS VIDEO BUTTON
            if st.button("Process Video for Human Detection"):
                if st.session_state.temp_video_path:
                    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                        try: os.remove(st.session_state.processed_video_path);
                        except: pass
                    st.session_state.processed_video_path = None

                    with st.spinner("Processing video... This may take some time."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix="_processed.mp4") as tmp_output_f:
                                output_video_path = tmp_output_f.name

                            # Call process_video_for_humans with the cached detector
                            process_video_for_humans(
                                human_detector, # The cached HumanDetector instance
                                st.session_state.temp_video_path,
                                output_video_path
                                # ROI is not yet passed to process_video_for_humans
                                # detection_threshold is also using default from process_video_for_humans
                            )
                            st.session_state.processed_video_path = output_video_path
                            st.success("Video processing complete!")
                        except Exception as e:
                            st.error(f"Error during video processing: {e}")
                            if 'output_video_path' in locals() and os.path.exists(output_video_path):
                                try: os.remove(output_video_path)
                                except: pass
                else: st.warning("No video uploaded to process.")
        else: st.error("Temporary video file is missing. Please re-upload.")

        # DISPLAY PROCESSED VIDEO
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            st.subheader("Processed Video with Human Detection")
            st.video(st.session_state.processed_video_path)
    else: # No file uploaded, cleanup logic
        if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            try: os.remove(st.session_state.temp_video_path)
            except: pass
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            try: os.remove(st.session_state.processed_video_path)
            except: pass
        st.session_state.temp_video_path = None
        st.session_state.processed_video_path = None
        st.session_state.roi_coords = None
        st.session_state.uploaded_file_name = None
        st.session_state.selected_frame_index = 0

if __name__ == "__main__":
    main()
