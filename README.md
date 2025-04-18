# 🐔 Chicken Detector & Behavior Monitoring Dashboard
<img width="100%" alt="image" src="https://github.com/user-attachments/assets/c4c83af3-657d-423d-b288-3a16a582a96a" />


This Streamlit-powered computer vision app detects, tracks, and analyzes the behavior of individual chickens in video footage using two custom-trained YOLOv8 models and a feature re-identification (ReID) pipeline.

---

## What It Does

- **Chicken Detection (YOLOv8):** Identifies chickens in uploaded videos.
- **Chicken Tracking (Deep Re-ID):** Uses cosine similarity and visual appearance features to consistently follow the same bird across frames.
- **Behavior Classification:** A second custom trained YOLOv8 model identifies behaviors such as:
  - Feeding
  - Resting
  - Panic movement
  - Pecking
  - Drinking
  - Walking
- **Real-Time UI:** Live dual-view dashboard with side-by-side tracking and detailed behavior analysis.
- **Visual Alerts:** Highlights when the selected chicken exits the frame with a blinking warning.

---

## 🧠 How It Works

1. **Model Setup**:
   - `best.pt`: Detection model for identifying all chickens.
   - `best_seg.pt`: Behavior classification model focused on the selected chicken.
   - `torchreid`: Used to re-identify the selected chicken across frames using visual feature vectors.

2. **Chicken Selection**:
   - The app extracts thumbnails of each detected chicken from the first frame.
   - The user manually selects one for detailed behavior tracking.

3. **Tracking Logic**:
   - Cosine similarity is computed between the selected chicken’s feature vector and crops in subsequent frames.
   - The most similar crop is considered the same chicken and highlighted.

4. **Behavior Analysis**:
   - For the matched chicken, the second YOLOv8 behaviour classification model is applied.
   - Predicted behavior is displayed in a labeled bounding box overlay.
