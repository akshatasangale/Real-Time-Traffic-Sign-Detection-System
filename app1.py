import cv2
import torch
import pyttsx3
import os

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Load YOLOv5 model from GitHub (NOT local)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# Video path
video_path = r"C:\Users\sanga\Downloads\Traffic-Sign-Recognition\vedio2.mp4"

# Check if video exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished or error in frame capture.")
        break

    # Inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        if row['confidence'] > 0.5:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak object name
            object_name = row['name']
            engine.say(object_name)
            engine.runAndWait()

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
