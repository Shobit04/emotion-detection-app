

# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image as PILImage
# import google.generativeai as genai
# import tempfile
# import os
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# from gtts import gTTS

# # ─────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────
# genai.configure(api_key="AIzaSyBgVc_3zL38oylAm2ejQoPN1jKO5KKr4fg")  # Replace with your actual Gemini API key
# gemini = genai.GenerativeModel("gemini-1.5-flash")
# model = YOLO("yolo11n-seg.pt")  # Path to YOLOv11n segmentation model

# # ─────────────────────────────────────
# # TEXT-TO-SPEECH FUNCTION
# # ─────────────────────────────────────
# def speak_text(text):
#     tts = gTTS(text)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         tts.save(f.name)
#         audio_file = f.name
#     audio_bytes = open(audio_file, 'rb').read()
#     st.audio(audio_bytes, format='audio/mp3')

# # ─────────────────────────────────────
# # UI: Streamlit Layout
# # ─────────────────────────────────────
# st.set_page_config(layout="wide")
# st.title("🎥 Instance Segmentation : Emotion Detection")

# mode = st.radio("Choose input method", ["📁 Upload Video", "📷 Use Webcam"])
# frame_rate = st.slider("📸 Frame Sampling Rate (frames/sec)", 1, 10, 1)
# custom_prompt = st.text_input("💬 Ask me", value="You are an expert in behavioral psychology. Observe the persons body language, facial expression, and context to describe their emotion using a rich vocabulary, not just basic moods like happy or sad.")
# tts_enabled = st.checkbox("🔊 Enable Text-to-Speech for Emotion Feedback")
# run_button = st.button("▶️ Start")

# emotion_results = []

# # ─────────────────────────────────────
# # Function: Extract Frames from Video
# # ─────────────────────────────────────
# def extract_frames(video_path, frame_rate=1):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     interval = max(fps // frame_rate, 1)
#     count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % interval == 0:
#             frames.append(frame)
#         count += 1
#     cap.release()
#     return frames

# # ─────────────────────────────────────
# # Video Source: Upload or Webcam
# # ─────────────────────────────────────
# frames = []
# if mode == "📁 Upload Video":
#     uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
#     if uploaded_file and run_button:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
#         st.video(video_path)
#         frames = extract_frames(video_path, frame_rate)
#         st.success(f"✅ Extracted {len(frames)} frame(s).")

# elif mode == "📷 Use Webcam":
#     if run_button:
#         cap = cv2.VideoCapture(0)
#         st.info("Recording from webcam... Press 'Stop' to process.")
#         stop_button = st.button("⏹️ Stop Webcam")
#         count = 0
#         while cap.isOpened() and not stop_button and count < 20:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if count % frame_rate == 0:
#                 frames.append(frame)
#                 st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"📷 Frame {count}")
#             count += 1
#         cap.release()
#         st.success(f"✅ Captured {len(frames)} frame(s) from webcam.")

# # ─────────────────────────────────────
# # Inference & Emotion Detection
# # ─────────────────────────────────────
# if frames:
#     emotion_counter = Counter()
#     for i, frame in enumerate(frames):
#         st.markdown(f"## 📷 Frame {i+1}")
#         results = model.predict(frame, conf=0.5, task='segment')[0]
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, caption="Original Frame", channels="RGB")

#         if not results.masks:
#             st.warning("No masks detected.")
#             continue

#         for j, mask in enumerate(results.masks.data):
#             cls_id = int(results.boxes.cls[j].item())
#             cls_name = results.names[cls_id]
#             if cls_name != "person":
#                 continue

#             mask_np = mask.cpu().numpy()
#             mask_resized = cv2.resize(mask_np, (frame_rgb.shape[1], frame_rgb.shape[0]))
#             mask_3d = np.stack([mask_resized] * 3, axis=-1)
#             person_crop = np.where(mask_3d > 0.5, frame_rgb, 0).astype(np.uint8)
#             pil_crop = PILImage.fromarray(person_crop)
#             st.image(pil_crop, caption="Detected Person")

#             try:
#                 response = gemini.generate_content([custom_prompt, pil_crop])
#                 emotion = response.text.strip()
#                 emotion_results.append({"frame": i + 1, "emotion": emotion})
#                 emotion_counter[emotion] += 1
#                 st.success(f"🧠 Emotion: {emotion}")
#                 if tts_enabled:
#                     speak_text(f"The detected emotion is {emotion}")
#             except Exception as e:
#                 st.error(f"Gemini Error: {e}")

# # ─────────────────────────────────────
# # Summary Pie Chart & CSV Download
# # ─────────────────────────────────────
# if emotion_results:
#     st.markdown("## 📊 Summary")
#     df = pd.DataFrame(emotion_results)
#     st.dataframe(df)

#     # Pie chart
#     fig, ax = plt.subplots()
#     labels = list(emotion_counter.keys())
#     values = list(emotion_counter.values())
#     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
#     ax.axis('equal')
#     st.pyplot(fig)

#     # Download CSV
#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button("📄 Download Emotion Report (CSV)", csv, "emotion_report.csv", "text/csv")



# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image as PILImage
# import google.generativeai as genai
# import tempfile
# import os
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# from gtts import gTTS
# from dotenv import load_dotenv


# # ─────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)
# gemini = genai.GenerativeModel("gemini-1.5-flash")
# model = YOLO("yolo11n-seg.pt")  # Path to YOLOv11n segmentation model

# # ─────────────────────────────────────
# # TEXT-TO-SPEECH FUNCTION
# # ─────────────────────────────────────
# def speak_text(text):
#     tts = gTTS(text)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         tts.save(f.name)
#         audio_file = f.name
#     audio_bytes = open(audio_file, 'rb').read()
#     st.audio(audio_bytes, format='audio/mp3')

# # ─────────────────────────────────────
# # UI: Streamlit Layout
# # ─────────────────────────────────────
# st.set_page_config(layout="wide")
# st.title("🎥 Instance Segmentation : Emotion Detection")

# mode = st.radio("Choose input method", ["📁 Upload Video", "📷 Use Webcam"])
# frame_rate = st.slider("📸 Frame Sampling Rate (frames/sec)", 1, 10, 1)
# custom_prompt = st.text_input("💬 Ask me", value="You are an expert in behavioral psychology. Observe the person's body language, facial expression, and context to describe their emotion using a rich vocabulary, not just basic moods like happy or sad.")
# tts_enabled = st.checkbox("🔊 Enable Text-to-Speech for Emotion Feedback")
# run_button = st.button("▶️ Start")

# emotion_results = []

# # ─────────────────────────────────────
# # Function: Extract Frames from Video
# # ─────────────────────────────────────
# def extract_frames(video_path, frame_rate=1):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     interval = max(fps // frame_rate, 1)
#     count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % interval == 0:
#             frames.append(frame)
#         count += 1
#     cap.release()
#     return frames

# # ─────────────────────────────────────
# # Video Source: Upload or Webcam
# # ─────────────────────────────────────
# frames = []
# if mode == "📁 Upload Video":
#     uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
#     if uploaded_file and run_button:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
#         st.video(video_path)
#         frames = extract_frames(video_path, frame_rate)
#         st.success(f"✅ Extracted {len(frames)} frame(s).")

# elif mode == "📷 Use Webcam":
#     if run_button:
#         cap = cv2.VideoCapture(0)
#         st.info("Recording from webcam... Press 'Stop' to process.")
#         stop_button = st.button("⏹️ Stop Webcam")
#         count = 0
#         while cap.isOpened() and not stop_button and count < 20:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if count % frame_rate == 0:
#                 frames.append(frame)
#                 st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"📷 Frame {count}")
#             count += 1
#         cap.release()
#         st.success(f"✅ Captured {len(frames)} frame(s) from webcam.")

# # ─────────────────────────────────────
# # Inference & Emotion Detection
# # ─────────────────────────────────────
# if frames:
#     emotion_counter = Counter()
#     for i, frame in enumerate(frames):
#         st.markdown(f"## 📷 Frame {i+1}")
#         results = model.predict(frame, conf=0.5, task='segment')[0]
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(frame_rgb, caption="Original Frame", channels="RGB")

#         if not results.masks:
#             st.warning("No masks detected.")
#             continue

#         for j, mask in enumerate(results.masks.data):
#             cls_id = int(results.boxes.cls[j].item())
#             cls_name = results.names[cls_id]
#             if cls_name != "person":
#                 continue

#             mask_np = mask.cpu().numpy()
#             mask_resized = cv2.resize(mask_np, (frame_rgb.shape[1], frame_rgb.shape[0]))
#             mask_3d = np.stack([mask_resized] * 3, axis=-1)
#             person_crop = np.where(mask_3d > 0.5, frame_rgb, 0).astype(np.uint8)
#             pil_crop = PILImage.fromarray(person_crop)
#             st.image(pil_crop, caption="Detected Person")

#             try:
#                 response = gemini.generate_content([custom_prompt, pil_crop])
#                 emotion = response.text.strip()

#                 # Summary word for emotion
#                 overall_emotion = "happy" if "happy" in emotion.lower() else \
#                   "sad" if "sad" in emotion.lower() else \
#                   "angry" if "angry" in emotion.lower() else \
#                   "shocked" if "shocked" in emotion.lower() else \
#                   "excited" if "excited" in emotion.lower() else \
#                   "bored" if "bored" in emotion.lower() else \
#                   "focused" if "focused" in emotion.lower() else \
#                   "surprised" if "surprised" in emotion.lower() else \
#                   "neutral" if "neutral" in emotion.lower() else \
#                   "disgusted" if "disgusted" in emotion.lower() else \
#                   "fearful" if "fearful" in emotion.lower() else \
#                   "guilty" if "guilty" in emotion.lower() else \
#                   "proud" if "proud" in emotion.lower() else \
#                   "relaxed" if "relaxed" in emotion.lower() else \
#                   "ashamed" if "ashamed" in emotion.lower() else \
#                   "jealous" if "jealous" in emotion.lower() else \
#                   "grateful" if "grateful" in emotion.lower() else \
#                   "embarrassed" if "embarrassed" in emotion.lower() else \
#                   "confused" if "confused" in emotion.lower() else \
#                   "hopeful" if "hopeful" in emotion.lower() else \
#                   "lonely" if "lonely" in emotion.lower() else \
#                   "motivated" if "motivated" in emotion.lower() else \
#                   "content" if "content" in emotion.lower() else \
#                   "disappointed" if "disappointed" in emotion.lower() else \
#                   "resentful" if "resentful" in emotion.lower() else \
#                   "nostalgic" if "nostalgic" in emotion.lower() else \
#                   "insecure" if "insecure" in emotion.lower() else \
#                   "ambitious" if "ambitious" in emotion.lower() else \
#                   "anxious" if "anxious" in emotion.lower() else \
#                   "overwhelmed" if "overwhelmed" in emotion.lower() else \
#                   "grief" if "grief" in emotion.lower() else \
#                   "indifferent" if "indifferent" in emotion.lower() else \
#                   "calm" if "calm" in emotion.lower() else \
#                   "suspicious" if "suspicious" in emotion.lower() else \
#                   "inlove" if "in love" in emotion.lower() else \
#                   "indignant" if "indignant" in emotion.lower() else \
#                   "emboldened" if "emboldened" in emotion.lower() else \
#                   "other"
#                 emotion_results.append({"frame": i + 1, "emotion": emotion})
#                 emotion_counter[overall_emotion] += 1
#                 st.success(f"🧠 Emotion: {emotion} (📊 Summary: {overall_emotion})")
#                 if tts_enabled:
#                     speak_text(f"The detected emotion is {emotion}")
#             except Exception as e:
#                 st.error(f"Gemini Error: {e}")

# # ─────────────────────────────────────
# # Summary Pie Chart & CSV Download
# # ─────────────────────────────────────
# if emotion_results:
#     st.markdown("## 📊 Summary")
#     df = pd.DataFrame(emotion_results)
#     st.dataframe(df)

#     # Pie chart with simplified labels
#     fig, ax = plt.subplots()
#     labels = list(emotion_counter.keys())
#     values = list(emotion_counter.values())
#     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
#     ax.axis('equal')
#     st.pyplot(fig)

#     # Download CSV
#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button("📄 Download Emotion Report (CSV)", csv, "emotion_report.csv", "text/csv")
# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image as PILImage
# import google.generativeai as genai
# import tempfile
# import os
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt
# from gtts import gTTS
# from dotenv import load_dotenv

# # ─────────────────────────────────────
# # PAGE CONFIG (must be first Streamlit call)
# # ─────────────────────────────────────
# st.set_page_config(layout="wide")

# # ─────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)
# gemini = genai.GenerativeModel("gemini-1.5-flash")
# model = YOLO("yolo11n-seg.pt")  # make sure this file lives next to app.py

# # ─────────────────────────────────────
# # TEXT-TO-SPEECH
# # ─────────────────────────────────────
# def speak_text(text):
#     tts = gTTS(text)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
#         tts.save(f.name)
#         audio_bytes = open(f.name, 'rb').read()
#     st.audio(audio_bytes, format='audio/mp3')

# # ─────────────────────────────────────
# # SESSION STATE FOR WEBCAM CAPTURE
# # ─────────────────────────────────────
# if "capturing" not in st.session_state:
#     st.session_state.capturing = False
# if "frames" not in st.session_state:
#     st.session_state.frames = []

# # ─────────────────────────────────────
# # UI LAYOUT
# # ─────────────────────────────────────
# st.title("🎥 Instance Segmentation : Emotion Detection")

# mode = st.radio("Choose input method", ["📁 Upload Video", "📷 Use Webcam"])
# frame_rate = st.slider("📸 Frame Sampling Rate (frames/sec)", 1, 10, 1)
# frame_limit = st.selectbox("Select the number of frames to generate", [10, 20])
# custom_prompt = st.text_input(
#     "💬 Ask me",
#     value=( 
#         "You are an expert in behavioral psychology. Observe the person's "
#         "body language, facial expression, and context to describe their "
#         "emotion using a rich vocabulary, not just basic moods."
#     )
# )
# tts_enabled = st.checkbox("🔊 Enable Text-to-Speech for Emotion Feedback")

# # ─────────────────────────────────────
# # FRAME EXTRACTION FUNCTION
# # ─────────────────────────────────────
# def extract_frames(video_path, frame_rate):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) or frame_rate
#     interval = max(fps // frame_rate, 1)
#     frames, count = [], 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % interval == 0:
#             frames.append(frame)
#         count += 1
#     cap.release()
#     return frames

# # ─────────────────────────────────────
# # UPLOAD VIDEO BRANCH
# # ─────────────────────────────────────
# if mode == "📁 Upload Video":
#     uploaded_file = st.file_uploader("Upload your video", type=["mp4","avi","mov"])
#     if uploaded_file and st.button("▶️ Process Video"):
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         path = tfile.name
#         st.video(path)
#         st.session_state.frames = extract_frames(path, frame_rate)
#         st.success(f"✅ Extracted {len(st.session_state.frames)} frames.")

# # ─────────────────────────────────────
# # WEBCAM BRANCH
# # ─────────────────────────────────────
# else:
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("▶️ Start Webcam"):
#             st.session_state.capturing = True
#             st.session_state.frames = []
#     with col2:
#         if st.button("⏹️ Stop Webcam"):
#             st.session_state.capturing = False

#     # Capturing frames from the webcam
#     if st.session_state.capturing:
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         cap.release()

#         # If the frame is successfully captured and the frame limit has not been reached
#         if ret and len(st.session_state.frames) < frame_limit:
#             st.session_state.frames.append(frame)
#         if ret:
#             st.image(frame[:,:,::-1], caption=f"Capturing #{len(st.session_state.frames)} frames")

#     elif st.session_state.frames:
#         st.success(f"✅ Captured {len(st.session_state.frames)} frames.")
#         # Automatically stop capturing once the limit is reached
#         if len(st.session_state.frames) >= frame_limit:
#             st.session_state.capturing = False
#             st.warning(f"Maximum limit of {frame_limit} frames reached. Stopping webcam capture.")

# # ─────────────────────────────────────
# # INFERENCE & EMOTION DETECTION
# # ─────────────────────────────────────
# frames = st.session_state.frames
# if frames and not st.session_state.capturing:
#     emotion_results, emotion_counter = [], Counter()
#     for idx, frame in enumerate(frames, 1):
#         st.markdown(f"## 📷 Frame {idx}")
#         res = model.predict(frame, conf=0.5, task='segment')[0]
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         st.image(rgb, caption="Original Frame", channels="RGB")

#         if not res.masks:
#             st.warning("No masks detected.")
#             continue

#         # crop & analyze each person
#         for j, mask in enumerate(res.masks.data):
#             cls = int(res.boxes.cls[j].item())
#             if res.names[cls] != "person":
#                 continue
#             mnp = mask.cpu().numpy()
#             mr = cv2.resize(mnp, (rgb.shape[1], rgb.shape[0]))
#             m3 = np.stack([mr]*3, axis=-1)
#             crop = np.where(m3>0.5, rgb, 0).astype(np.uint8)
#             st.image(crop, caption="Detected Person")

#             try:
#                 resp = gemini.generate_content([custom_prompt, PILImage.fromarray(crop)])
#                 emotion = resp.text.strip()
#             except Exception as e:
#                 st.error(f"Gemini Error: {e}")
#                 emotion = "error"

#             # Extended emotions
#             overall_emotion = (
#                 "happy" if "happy" in emotion.lower() else
#                 "sad" if "sad" in emotion.lower() else
#                 "angry" if "angry" in emotion.lower() else
#                 "shocked" if "shocked" in emotion.lower() else
#                 "excited" if "excited" in emotion.lower() else
#                 "bored" if "bored" in emotion.lower() else
#                 "focused" if "focused" in emotion.lower() else
#                 "surprised" if "surprised" in emotion.lower() else
#                 "neutral" if "neutral" in emotion.lower() else
#                 "disgusted" if "disgusted" in emotion.lower() else
#                 "fearful" if "fearful" in emotion.lower() else
#                 "guilty" if "guilty" in emotion.lower() else
#                 "proud" if "proud" in emotion.lower() else
#                 "relaxed" if "relaxed" in emotion.lower() else
#                 "ashamed" if "ashamed" in emotion.lower() else
#                 "jealous" if "jealous" in emotion.lower() else
#                 "grateful" if "grateful" in emotion.lower() else
#                 "embarrassed" if "embarrassed" in emotion.lower() else
#                 "confused" if "confused" in emotion.lower() else
#                 "hopeful" if "hopeful" in emotion.lower() else
#                 "lonely" if "lonely" in emotion.lower() else
#                 "motivated" if "motivated" in emotion.lower() else
#                 "content" if "content" in emotion.lower() else
#                 "disappointed" if "disappointed" in emotion.lower() else
#                 "resentful" if "resentful" in emotion.lower() else
#                 "nostalgic" if "nostalgic" in emotion.lower() else
#                 "insecure" if "insecure" in emotion.lower() else
#                 "ambitious" if "ambitious" in emotion.lower() else
#                 "anxious" if "anxious" in emotion.lower() else
#                 "overwhelmed" if "overwhelmed" in emotion.lower() else
#                 "grief" if "grief" in emotion.lower() else
#                 "indifferent" if "indifferent" in emotion.lower() else
#                 "calm" if "calm" in emotion.lower() else
#                 "suspicious" if "suspicious" in emotion.lower() else
#                 "inlove" if "in love" in emotion.lower() else
#                 "indignant" if "indignant" in emotion.lower() else
#                 "emboldened" if "emboldened" in emotion.lower() else
#                 "other"
#             )

#             emotion_results.append({"frame": idx, "emotion": emotion})
#             emotion_counter[overall_emotion] += 1
#             st.success(f"🧠 {emotion}  —  ({overall_emotion})")
#             if tts_enabled:
#                 speak_text(emotion)

#     # summary chart + CSV
#     if emotion_results:
#         st.markdown("## 📊 Summary")
#         df = pd.DataFrame(emotion_results)
#         st.dataframe(df)
#         fig, ax = plt.subplots()
#         ax.pie(emotion_counter.values(), labels=emotion_counter.keys(),
#                autopct="%1.1f%%", startangle=90)
#         ax.axis("equal")
#         st.pyplot(fig)
#         csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button("📄 Download CSV", csv, "report.csv", "text/csv")

