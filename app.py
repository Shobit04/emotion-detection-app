



import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image as PILImage
import google.generativeai as genai
import tempfile
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from gtts import gTTS

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
genai.configure(api_key="AIzaSyBgVc_3zL38oylAm2ejQoPN1jKO5KKr4fg")  # Replace with your actual Gemini API key
gemini = genai.GenerativeModel("gemini-1.5-flash")
model = YOLO("yolo11n-seg.pt")  # Path to YOLOv11n segmentation model

# ─────────────────────────────────────
# TEXT-TO-SPEECH FUNCTION
# ─────────────────────────────────────
def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio_file = f.name
    audio_bytes = open(audio_file, 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')

# ─────────────────────────────────────
# UI: Streamlit Layout
# ─────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🎥 Instance Segmentation : Emotion Detection")

mode = st.radio("Choose input method", ["📁 Upload Video", "📷 Use Webcam"])
frame_rate = st.slider("📸 Frame Sampling Rate (frames/sec)", 1, 10, 1)
custom_prompt = st.text_input("💬 Ask me", value="You are an expert in behavioral psychology. Observe the persons body language, facial expression, and context to describe their emotion using a rich vocabulary, not just basic moods like happy or sad.")
tts_enabled = st.checkbox("🔊 Enable Text-to-Speech for Emotion Feedback")
run_button = st.button("▶️ Start")

emotion_results = []
emotion_counter = Counter()

# ─────────────────────────────────────
# Function: Extract Frames from Video
# ─────────────────────────────────────
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(fps // frame_rate, 1)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# ─────────────────────────────────────
# Function: Classify Extended Emotions
# ─────────────────────────────────────
def classify_extended_emotion(emotion_text):
    emotion_text = emotion_text.lower()
    
    if "happy" in emotion_text:
        return "happy"
    elif "sad" in emotion_text:
        return "sad"
    elif "angry" in emotion_text:
        return "angry"
    elif "shocked" in emotion_text:
        return "shocked"
    elif "excited" in emotion_text:
        return "excited"
    elif "bored" in emotion_text:
        return "bored"
    elif "focused" in emotion_text:
        return "focused"
    elif "surprised" in emotion_text:
        return "surprised"
    elif "neutral" in emotion_text:
        return "neutral"
    elif "disgusted" in emotion_text:
        return "disgusted"
    elif "fearful" in emotion_text:
        return "fearful"
    elif "guilty" in emotion_text:
        return "guilty"
    elif "proud" in emotion_text:
        return "proud"
    elif "relaxed" in emotion_text:
        return "relaxed"
    elif "ashamed" in emotion_text:
        return "ashamed"
    elif "jealous" in emotion_text:
        return "jealous"
    elif "grateful" in emotion_text:
        return "grateful"
    elif "embarrassed" in emotion_text:
        return "embarrassed"
    elif "confused" in emotion_text:
        return "confused"
    elif "hopeful" in emotion_text:
        return "hopeful"
    elif "lonely" in emotion_text:
        return "lonely"
    elif "motivated" in emotion_text:
        return "motivated"
    elif "content" in emotion_text:
        return "content"
    elif "disappointed" in emotion_text:
        return "disappointed"
    elif "resentful" in emotion_text:
        return "resentful"
    elif "nostalgic" in emotion_text:
        return "nostalgic"
    elif "insecure" in emotion_text:
        return "insecure"
    elif "ambitious" in emotion_text:
        return "ambitious"
    elif "anxious" in emotion_text:
        return "anxious"
    elif "overwhelmed" in emotion_text:
        return "overwhelmed"
    elif "grief" in emotion_text:
        return "grief"
    elif "indifferent" in emotion_text:
        return "indifferent"
    elif "calm" in emotion_text:
        return "calm"
    elif "suspicious" in emotion_text:
        return "suspicious"
    elif "in love" in emotion_text:
        return "inlove"
    elif "indignant" in emotion_text:
        return "indignant"
    elif "emboldened" in emotion_text:
        return "emboldened"
    else:
        return "other"

# ─────────────────────────────────────
# Video Source: Upload or Webcam
# ─────────────────────────────────────
frames = []
if mode == "📁 Upload Video":
    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
    if uploaded_file and run_button:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)
        frames = extract_frames(video_path, frame_rate)
        st.success(f"✅ Extracted {len(frames)} frame(s).")

elif mode == "📷 Use Webcam":
    if run_button:
        cap = cv2.VideoCapture(0)
        st.info("Recording from webcam... Press 'Stop' to process.")
        stop_button = st.button("⏹️ Stop Webcam")
        count = 0
        while cap.isOpened() and not stop_button and count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_rate == 0:
                frames.append(frame)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"📷 Frame {count}")
            count += 1
        cap.release()
        st.success(f"✅ Captured {len(frames)} frame(s) from webcam.")

# ─────────────────────────────────────
# Inference & Emotion Detection
# ─────────────────────────────────────
if frames:
    for i, frame in enumerate(frames):
        st.markdown(f"## 📷 Frame {i+1}")
        results = model.predict(frame, conf=0.5, task='segment')[0]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Original Frame", channels="RGB")

        if not results.masks:
            st.warning("No masks detected.")
            continue

        for j, mask in enumerate(results.masks.data):
            cls_id = int(results.boxes.cls[j].item())
            cls_name = results.names[cls_id]
            if cls_name != "person":
                continue

            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (frame_rgb.shape[1], frame_rgb.shape[0]))
            mask_3d = np.stack([mask_resized] * 3, axis=-1)
            person_crop = np.where(mask_3d > 0.5, frame_rgb, 0).astype(np.uint8)
            pil_crop = PILImage.fromarray(person_crop)
            st.image(pil_crop, caption="Detected Person")

            try:
                response = gemini.generate_content([custom_prompt, pil_crop])
                emotion = response.text.strip()
                overall_emotion = classify_extended_emotion(emotion)
                
                emotion_results.append({"frame": i + 1, "emotion": emotion, "category": overall_emotion})
                emotion_counter[overall_emotion] += 1
                
                st.success(f"🧠 {emotion}  — (📊 Summary: {overall_emotion})")
                
                if tts_enabled:
                    speak_text(emotion)
            except Exception as e:
                st.error(f"Gemini Error: {e}")

# ─────────────────────────────────────
# Summary Pie Chart & CSV Download
# ─────────────────────────────────────
if emotion_results:
    st.markdown("## 📊 Summary")
    df = pd.DataFrame(emotion_results)
    st.dataframe(df)

    # Pie chart for emotion categories
    fig, ax = plt.subplots()
    labels = list(emotion_counter.keys())
    values = list(emotion_counter.values())
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Emotion Report (CSV)", csv, "emotion_report.csv", "text/csv")