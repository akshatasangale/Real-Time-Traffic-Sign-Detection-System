## For Realtime detection through laptop's camera
import cv2
import torch
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speed of speech (default is 200)
engine.setProperty('volume', 1.0)  # Set volume (from 0.0 to 1.0)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# Initialize the camera (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # YOLOv5 inference on the captured frame
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame

    # Draw detections on the frame
    for _, row in detections.iterrows():
        # Only process detections with confidence > 0.6 (50%)
        if row['confidence'] > 0.7:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Use text-to-speech to announce the detected object
            object_name = row['name']
            engine.say(object_name)
            engine.runAndWait()

    # Display the resulting frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Press 'q' to quit the video capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

##-------------------------------For Inbuilt Video--------------------

# import cv2
# import torch
# import pyttsx3
# import os

# # Initialize TTS engine
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Set speed of speech (default is 200)
# engine.setProperty('volume', 1.0)  # Set volume (from 0.0 to 1.0)

# # Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# # Provide the path to your video file
# video_path = "F:/Traffic_Sign/Traffic-Sign-Recognition/video.mp4"  # Replace this with your actual path

# # Check if the video file exists
# if not os.path.exists(video_path):
#     print(f"Error: Video file not found at {video_path}")
#     exit()

# # Initialize video capture with the video file
# cap = cv2.VideoCapture(video_path)

# # Check if the video was opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video file.")
#     exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame or end of video reached.")
#         break

#     # YOLOv5 inference on the captured frame
#     results = model(frame)
#     detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame

#     # Draw detections on the frame
#     for _, row in detections.iterrows():
#         # Only process detections with confidence > 0.5 (50%)
#         if row['confidence'] > 0.5:
#             x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             label = f"{row['name']} {row['confidence']:.2f}"
            
#             # Draw rectangle and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#             # Use text-to-speech to announce the detected object
#             object_name = row['name']
#             engine.say(object_name)
#             engine.runAndWait()

#     # Display the resulting frame
#     cv2.imshow("YOLOv5 Object Detection", frame)

#     # Press 'q' to quit the video capture
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close any OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
