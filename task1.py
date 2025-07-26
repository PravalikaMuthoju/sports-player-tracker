import streamlit as st
import cv2
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import os
import tempfile
import time

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Player Re-Identification Demo", layout="wide")

# Verify transformers and timm installation
try:
    import transformers
    import timm
except ImportError as e:
    st.error(f"Required library not installed: {str(e)}. Please install 'transformers' and 'timm' using requirements.txt.")
    st.stop()

@st.cache_resource
def load_model():
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def process_frame(frame, processor, model, confidence_threshold=0.7):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([frame_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]
    person_boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "person" and score.item() > confidence_threshold:
            person_boxes.append((box.tolist(), score.item()))
    return person_boxes

def process_video(video_path, output_path, processor, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 1280  # Fixed to 720p width
    height = int(720 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        st.error(f"Failed to initialize VideoWriter for {output_path}")
        cap.release()
        return False
    frame_count = 0
    player_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    max_processing_time = 300  # 5-minute timeout in seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        person_boxes = process_frame(frame, processor, model)
        for box, score in person_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {player_id} ({score:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            player_id += 1
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)
        frame_count += 1
        progress = frame_count / total_frames if total_frames > 0 else 0
        st.progress(min(progress, 1.0))  # Update progress bar
        if time.time() - start_time > max_processing_time:
            st.error("Processing timed out after 5 minutes. Video may be too long or complex.")
            break
    cap.release()
    out.release()
    return frame_count > 0  # Return True if at least one frame was processed

st.title("Player Re-Identification in Sports Footage")
st.write("Upload a sports video to detect and track players using a Hugging Face DETR model. Recommended format: 720p, H.264 codec, under 100MB.")
processor, model = load_model()
if processor is None or model is None:
    st.stop()
uploaded_file = st.file_uploader("Upload a video file (e.g., broadcast.mp4 or tacticam.mp4)", type=["mp4"])
if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}, size: {uploaded_file.size} bytes, type: {uploaded_file.type}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    st.write(f"Temporary file path: {video_path}")
    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
    with st.spinner("Processing video..."):
        success = process_video(video_path, output_path, processor, model)
    if success:
        st.success("Video processed successfully!")
        st.video(output_path)
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="output.mp4",
                mime="video/mp4"
            )
    else:
        st.error("Video processing failed or timed out. Please try a shorter video or check compatibility.")
    # Clean up temporary files
    os.unlink(video_path)
    if os.path.exists(output_path):
        os.unlink(output_path)