import cv2
import streamlit as st
from pathlib import Path
import sys
import threading
import time
import queue 
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import math



# ==============================
# 1️⃣ FILE PATH & MODEL CONFIGURATION
# ==============================

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Add the root path to sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

ROOT = ROOT.relative_to(Path.cwd())

# Video & Model Sources
VIDEO = 'Chicken Detector'
UPLOAD = 'Behaviour Analysis'
MULTI_STREAM = 'Multi-Stream'
SOURCES_LIST = [VIDEO, UPLOAD]

MODEL_NAMES = ["weights/best.pt", "weights/yolo11n-seg.pt"]
VIDEOS_DICT = {
    "Sample video 1": "videos/chicken_video1.mp4",
    "Sample video 2": "videos/chicken_video2.mp4",
    "Sample video 3": "videos/test_clip.mp4"
}

# Model Path
MODEL_DIR = ROOT / 'weights'
#DETECTION_MODEL = MODEL_DIR / 'best_chicken_detector.pt'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Load YOLO Model
model = YOLO(str(DETECTION_MODEL))

# ✅ Create a Queue for Thread-Safe Communication
results_queue = queue.Queue()

# ==============================
# 2️⃣ STREAMLIT UI SETUP
# ==============================

st.set_page_config(layout="wide")  # Full-width UI layout
st.title("🐔 Chicken Detection & Multi-Stream Behavior Analysis")

# Define Source Options
OVERVIEW = "Overview & Instructions"
VIDEO = "Chicken Detector"
UPLOAD = "Behaviour Analysis"
MULTI_STREAM = "Multi-Stream"
SOURCES_LIST = [VIDEO, UPLOAD]

# Sidebar source selection (default: VIDEO)
source = st.sidebar.selectbox("Choose an option", SOURCES_LIST, index=0)

# ==============================
# 🎯 Dynamic Subtitle Based on Selection
# ==============================
if source == VIDEO:
    st.subheader("Chicken Detection Mode")
elif source == UPLOAD:
    st.subheader("Behaviour Analysis Mode — Analyze Individual Chickens")

# Live Count Sidebar Placeholders
live_chicken_count1 = st.sidebar.empty()
live_chicken_count2 = st.sidebar.empty()


# ==============================
# 3️⃣ VIDEO UPLOAD & SELECTION
# ==============================

track_history = defaultdict(lambda: [])

# 📂 Sidebar widgets first
video_option = st.sidebar.radio("Select Sample Video", list(VIDEOS_DICT.keys()), index=0)
uploaded_video = st.sidebar.file_uploader("Or Upload Your Own Video", type=["mp4", "avi", "mov"])

# 🧠 Initialize upload flag
if "uploaded_video_saved" not in st.session_state:
    st.session_state["uploaded_video_saved"] = False

# 📍 Determine final video path
video_path = None

# ✅ Handle uploaded video first
if uploaded_video and not st.session_state["uploaded_video_saved"]:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.session_state["uploaded_video_saved"] = True
    st.rerun()

elif uploaded_video:
    video_path = "uploaded_video.mp4"

# ✅ Fallback to sample video if nothing uploaded
elif video_option:
    video_path = str(VIDEOS_DICT.get(video_option, "videos/chicken_video1.mp4"))



import queue
from threading import Thread
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

import torchreid
from torchreid.utils import FeatureExtractor

# ============================================
# Setup TorchReID
# ============================================
extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path=None,
    device='cpu'
)

def extract_reid_feature(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))
    feature = extractor(img)
    return feature[0]

# ============================================
# Main Streamlit + YOLOv8 App
# ============================================

DETECTION_MODEL = MODEL_DIR / "best.pt"
DETAILED_MODEL_PATH = MODEL_DIR / "best_seg.pt"

model = YOLO(str(DETECTION_MODEL))
detailed_model = YOLO(str(DETAILED_MODEL_PATH))

# ============================================
# 🧹 Clear chicken selection if new video is selected
# ============================================

if "selected_chicken_data" not in st.session_state:
    st.session_state["selected_chicken_data"] = None
if "selected_chicken_iou_frames" not in st.session_state:
    st.session_state["selected_chicken_iou_frames"] = 0
if "selected_chicken_disappear_frames" not in st.session_state:
    st.session_state["selected_chicken_disappear_frames"] = 0
if "analyze_selected_chicken" not in st.session_state:
    st.session_state["analyze_selected_chicken"] = False
if "last_video_path" not in st.session_state:
    st.session_state["last_video_path"] = None

# 🧠 Detect if video changed
if (
    video_path != st.session_state["last_video_path"] and
    st.session_state["last_video_path"] is not None
):


    # Reset chicken tracking session state
    st.session_state["selected_chicken_data"] = None
    st.session_state["selected_chicken_iou_frames"] = 0
    st.session_state["selected_chicken_disappear_frames"] = 0
    st.session_state["analyze_selected_chicken"] = False

# ✅ Always update the last video path
st.session_state["last_video_path"] = video_path


if video_path and source == UPLOAD:
    st.markdown("Select a Chicken")

    cap = cv2.VideoCapture(video_path)
    ret, frame_selector = cap.read()
    cap.release()

    if ret:
        track_result = model.track(source=frame_selector, persist=True)[0]
        boxes = track_result.boxes

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            selected_data = st.session_state.get("selected_chicken_data")
            has_selection = selected_data is not None

            cols = st.columns(min(len(track_ids), 6))
            for i, col in enumerate(cols):
                
                x1, y1, x2, y2 = map(int, xyxy[i])
                crop = frame_selector[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                thumb = cv2.resize(crop, (120, 120))
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(thumb_rgb)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()

                is_selected = has_selection and np.allclose(xyxy[i], selected_data["box"])
                opacity = 1.0 if (not has_selection or is_selected) else 0.4
                text_color = "#00ff88" if is_selected else ("#000" if not has_selection else "#aaa")
                font_weight = "bold" if is_selected else "normal"

                with col:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <img src="data:image/jpeg;base64,{img_b64}" 
                                 style="width: 100%; border-radius: 8px; opacity: {opacity};" />
                            <p style="margin-top: 4px; color: {text_color}; font-weight: {font_weight};">
                                ID {track_ids[i]}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    button_key = f"btn_{track_ids[i]}"
                    if is_selected:
                        st.button("✅ Selected", key=button_key, disabled=True)
                    else:
                        if st.button(f"Select ID {track_ids[i]}", key=button_key):
                            feature = extract_reid_feature(crop)
                            st.session_state["selected_chicken_data"] = {
                                "box": xyxy[i].tolist(),
                                "feature": feature
                            }
                            st.session_state["selected_chicken_iou_frames"] = 0
                            st.session_state["selected_chicken_disappear_frames"] = 0
                            st.session_state["analyze_selected_chicken"] = True
                            st.rerun()
        else:
            st.warning("⚠️ No chickens detected in the first frame.")

    # === Tracker + Detail Views
    col1, col2 = st.columns(2)
    tracker_output = col1.empty()
    detailed_output = col2.empty()

    tracker_model_name = DETECTION_MODEL.name
    detailed_model_name = DETAILED_MODEL_PATH.name

    selected_data = st.session_state.get("selected_chicken_data")
    analyze = st.session_state.get("analyze_selected_chicken", False)

    if analyze and selected_data is not None:
        results_queue = queue.Queue()
        frame_count = 0
        blink_interval = 20

        def run_tracking_loop():
            results = model.track(source=video_path, stream=True, persist=True)
            for result in results:
                results_queue.put(result)

        thread = Thread(target=run_tracking_loop, daemon=True)
        thread.start()

        while thread.is_alive() or not results_queue.empty():
            while not results_queue.empty():
                result = results_queue.get()
                frame = result.orig_img.copy()
                tracker_frame = result.plot()
                boxes = result.boxes
                bottom_label = "No Behaviour Detected"

                fh, fw = frame.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(1.0, fh / 700)
                thickness = max(2, int(fh / 300))
                pad = int(fh * 0.02)
                missing_warning = False

                best_match = None
                best_sim = -1
                sim_threshold = 0.65

                chicken_count = 0  # Default

                if boxes and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    chicken_count = len(ids)

                    # 🐔 Bigger, left-aligned live chicken count
                    count_scale = scale * 1.6  # 60% larger text
                    count_thickness = thickness + 2

                    label_text = f"Live Count: {chicken_count} Chickens"
                    size = cv2.getTextSize(label_text, font, count_scale, count_thickness)[0]

                    tx = 20  # padding from left
                    ty = fh - 20  # padding from bottom

                    # Semi-transparent background box
                    overlay = tracker_frame.copy()
                    cv2.rectangle(overlay,
                                (tx - 10, ty - size[1] - 10),
                                (tx + size[0] + 10, ty + 10),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, tracker_frame, 0.4, 0, tracker_frame)

                    # Render bigger white text
                    cv2.putText(tracker_frame, label_text, (tx, ty),
                                font, count_scale, (255, 255, 255), count_thickness, lineType=cv2.LINE_AA)

        

                    last_feature = selected_data["feature"].reshape(1, -1)

                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        feature = extract_reid_feature(crop).reshape(1, -1)
                        sim = cosine_similarity(last_feature, feature)[0][0]

                        cv2.putText(tracker_frame, f"{sim:.2f}", (x1, y1 - 10),
                                    font, 0.5, (255, 255, 0), 1)

                        if sim > sim_threshold and sim > best_sim:
                            best_sim = sim
                            best_match = {
                                "index": i,
                                "box": box.tolist(),
                                "feature": feature[0]
                            }

                if best_match:
                    st.session_state["selected_chicken_data"] = best_match
                    st.session_state["selected_chicken_iou_frames"] += 1
                    st.session_state["selected_chicken_disappear_frames"] = 0

                    x1, y1, x2, y2 = map(int, best_match["box"])
                    label_text = "Selected"
                    label_size = cv2.getTextSize(label_text, font, scale, thickness)[0]
                    label_y = y2 + 30

                    for f in [tracker_frame, frame]:
                        cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.rectangle(f, (x1, label_y - label_size[1] - 5),
                                         (x1 + label_size[0] + 10, label_y + 5), (0, 255, 0), -1)
                        cv2.putText(f, label_text, (x1 + 5, label_y), font, scale, (0, 0, 0), thickness)

                    chicken_crop = frame[y1:y2, x1:x2]
                    if chicken_crop.shape[0] > 0 and chicken_crop.shape[1] > 0:
                        try:
                            detailed = detailed_model.predict(
                                source=chicken_crop,
                                conf=0.5,
                                stream=False,
                                verbose=False
                            )[0]

                            if detailed.boxes is not None and len(detailed.boxes.cls) > 0:
                                cls_idx = int(detailed.boxes.cls[0])
                                conf = float(detailed.boxes.conf[0])
                                class_name = detailed.names[cls_idx]
                                bottom_label = f"{class_name} ({conf:.2f})"

                            if detailed.masks is not None:
                                masks = detailed.masks.data.cpu().numpy()
                                for mask in masks:
                                    color = (0, 255, 0)
                                    ch, cw = chicken_crop.shape[:2]
                                    resized_mask = cv2.resize(mask.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
                                    color_mask = np.zeros_like(chicken_crop, dtype=np.uint8)
                                    for c in range(3):
                                        color_mask[:, :, c] = resized_mask * color[c]
                                    blended = cv2.addWeighted(chicken_crop, 1.0, color_mask, 0.5, 0)
                                    frame[y1:y2, x1:x2] = blended

                        except Exception as e:
                            print("🐔 Prediction failed:", e)

                else:
                    st.session_state["selected_chicken_iou_frames"] = 0
                    st.session_state["selected_chicken_disappear_frames"] += 1
                    missing_warning = True

                # 🔴 Blinking alert if chicken disappears
                if st.session_state["selected_chicken_disappear_frames"] > 10 and missing_warning:
                    if (frame_count // (blink_interval // 2)) % 2 == 0:
                        warning_text = "Selected chicken left the frame!"
                        
                        # 🔥 Make font larger and bolder
                        alert_scale = scale * 1.8
                        alert_thickness = thickness + 2
                        
                        size = cv2.getTextSize(warning_text, font, alert_scale, alert_thickness)[0]
                        tx = (fw - size[0]) // 2
                        ty = int(fh * 0.12)  # A bit lower than top

                        # 🔴 Bigger red background box
                        cv2.rectangle(frame, 
                                    (tx - 30, ty - size[1] - 20),
                                    (tx + size[0] + 30, ty + 20), 
                                    (0, 0, 255), -1)

                        # 📝 White bold text
                        cv2.putText(frame, warning_text, (tx, ty), font, alert_scale, (255, 255, 255), alert_thickness)


                # Bottom label on detailed frame
                if bottom_label:
                    bottom_scale = scale * 1.4       
                    bottom_thickness = thickness + 1 
                    size = cv2.getTextSize(bottom_label, font, bottom_scale, bottom_thickness)[0]
                    tx = (fw - size[0]) // 2
                    ty = fh - size[1] - pad

                    cv2.rectangle(frame, (tx - 20, ty - size[1] - 10),
                                (tx + size[0] + 20, ty + 10), (0, 0, 0), -1)

                    cv2.putText(frame, bottom_label, (tx, ty), font, bottom_scale, (0, 255, 0), bottom_thickness)



                tracker_output.image(
                    cv2.cvtColor(tracker_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption=f"MODEL: {tracker_model_name}"
                )

                detailed_output.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    caption=f"MODEL: {detailed_model_name}"
                )

                frame_count += 1
                time.sleep(0.03)

        st.success("✅ Finished video analysis.")



# ==============================================
# 4️⃣ HELPER FUNCTION: TRACKER WITH COUNT OVERLAY
# ==============================================

def run_tracker_in_thread(model_path, video_path, video_key):
    """ Runs YOLO Tracker in a separate thread, overlays count, and adds results to the queue. """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error loading {video_path}!")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None or frame.size == 0:
            continue  # Skip invalid frames

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Run YOLO tracking
        results = model.track(source=frame, persist=True, conf=0.5)

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # ✅ Dynamically scale font size based on video resolution
            font_scale = max(1, frame_height / 500)
            font_thickness = max(2, int(frame_height / 300))
            padding = int(frame_height * 0.02)

            count_text = f"Live Count: {len(track_ids)} Chickens"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(count_text, font, font_scale, font_thickness)[0]

            text_x = (frame_width - text_size[0]) // 2
            text_y = frame_height - text_size[1] - padding

            # ✅ Draw a semi-transparent rectangle
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # ✅ Display text
            cv2.putText(annotated_frame, count_text, (text_x, text_y), 
                        font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

            # ✅ Add results to the queue
            results_queue.put((video_key, annotated_frame, len(track_ids)))

        time.sleep(0.05)

    cap.release()

# ===========================
# 5️⃣ SINGLE VIDEO PROCESSING
# ===========================

if source in [VIDEO, UPLOAD] and video_path:
    
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error loading video!")
    else:
        frame_placeholder = st.empty()
        live_chicken_count = st.sidebar.empty()
        track_history = defaultdict(lambda: [])  # ✅ Restore tracking history

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection & tracking
            results = model.track(source=frame, persist=True, conf=0.5)

            if results[0].boxes.id is not None:
                # Get bounding boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize results
                annotated_frame = results[0].plot()

                # ✅ Restore tracking trails
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # Store (x, y) center points
                    if len(track) > 30:
                        track.pop(0)  # Keep track length reasonable

                    # Draw tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=2)

                # ✅ Overlay chicken count at the bottom of the frame
                frame_height, frame_width = annotated_frame.shape[:2]
                font_scale = max(1, frame_height / 500)
                font_thickness = max(2, int(frame_height / 300))
                padding = int(frame_height * 0.02)

                count_text = f"Live Count: {len(track_ids)} Chickens"
                text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                text_x = (frame_width - text_size[0]) // 2  # Center horizontally
                text_y = frame_height - text_size[1] - padding  # Position at the bottom

                # ✅ Draw a semi-transparent rectangle behind text
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 10), 
                              (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)

                # ✅ Blend for transparency
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

                # ✅ Display the count on the frame
                cv2.putText(annotated_frame, count_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

                # Convert BGR to RGB for Streamlit
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # ✅ Display the frame in Streamlit
                frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

                # ✅ Update sidebar live count
                live_chicken_count.markdown(f"**Live Count: {len(track_ids)} Chickens**")

            time.sleep(0.05)  # Small delay for smooth display

        cap.release()


# ===========================
# 6️⃣ MULTI-STREAM PROCESSING
# ===========================

elif source == MULTI_STREAM:
    st.subheader("📽️ Multi-Stream Tracking - Two Video Streams")
    col1, col2 = st.columns([2, 2])
    video_placeholder1 = col1.empty()
    video_placeholder2 = col2.empty()

    tracker_threads = []
    video_keys = ["video1", "video2"]

    for (video_name, video_path), model_path, video_key in zip(VIDEOS_DICT.items(), MODEL_NAMES, video_keys):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_path, video_path, video_key), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    while any(thread.is_alive() for thread in tracker_threads):
        while not results_queue.empty():
            fh, fw = frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(1.0, fh / 700)
            thickness = max(2, int(fh / 300))
            pad = int(fh * 0.02)

            video_key, annotated_frame, count = results_queue.get()
            if video_key == "video1":
                video_placeholder1.image(annotated_frame, channels="RGB", use_column_width=True)
            else:
                video_placeholder2.image(annotated_frame, channels="RGB", use_column_width=True)

        time.sleep(0.1)

    for thread in tracker_threads:
        thread.join()


st.markdown("Developed for Chicken Health Monitoring 🐔 using Computer Vision.")


