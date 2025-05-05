# process_utils.py

import torch
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv11 instance segmentation model
yolo_model = YOLO("yolov11n-seg.pt")  # Update if using different weights

# Load DeepSORT tracker
deep_sort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# Dummy Gemini emotion detection function (replace with your actual function)
def detect_emotion_from_prompt(prompt):
    # Replace this with actual Gemini API call
    return "happy"  # Placeholder

def process_frame(frame):
    """
    Processes a single frame:
    - Runs YOLO instance segmentation
    - Tracks objects with DeepSORT
    - Detects emotion from prompt
    - Returns updated frame and results
    """
    results = yolo_model.predict(frame, verbose=False)[0]

    bboxes = []
    confidences = []
    class_ids = []

    for r in results.boxes:
        bbox = r.xyxy.cpu().numpy()[0]
        conf = float(r.conf)
        cls = int(r.cls)
        bboxes.append(bbox)
        confidences.append(conf)
        class_ids.append(cls)

    if bboxes:
        bboxes_xywh = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            bbox_xywh = [((x1 + x2) / 2), ((y1 + y2) / 2), x2 - x1, y2 - y1]
            bboxes_xywh.append(bbox_xywh)

        outputs = deep_sort.update(
            np.array(bboxes_xywh),
            np.array(confidences),
            np.array(class_ids),
            frame,
        )
    else:
        outputs = []

    annotated_frame = frame.copy()
    for output in outputs:
        x1, y1, x2, y2, track_id = output
        prompt = f"A person doing something with ID {track_id}"
        emotion = detect_emotion_from_prompt(prompt)

        # Draw bounding box and label
        annotated_frame = cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        annotated_frame = cv2.putText(annotated_frame, f"{emotion}", (int(x1), int(y1)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    return annotated_frame, outputs
