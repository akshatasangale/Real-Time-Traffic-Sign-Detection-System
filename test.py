import cv2
import torch
import os
import pyttsx3

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # adjust speech rate if needed

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        print(f"Frame {frame_count} detections:\n", detections)

        # Draw bounding boxes and labels
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = str(row['name'])
            conf = float(row['confidence'])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Every 30 frames, speak out the detected sign names (if any)
        if frame_count % 30 == 0:
            spoken_labels = set()
            for _, row in detections.iterrows():
                label = str(row['name'])
                if label not in spoken_labels:
                    engine.say(label)
                    spoken_labels.add(label)
            # Run the speech queue and block until complete (if too blocking, consider alternative threading)
            engine.runAndWait()

        cv2.imshow('Traffic Sign Detection', frame)
        frame_count += 1

        # Press 'q' to quit video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    img = cv2.imread(image_path)
    results = model(img)
    detections = results.pandas().xyxy[0]
    print("Image detections:\n", detections)

    spoken_labels = set()
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = str(row['name'])
        conf = float(row['confidence'])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if label not in spoken_labels:
            engine.say(label)
            spoken_labels.add(label)
            
    engine.runAndWait()

    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# # Uncomment one of the following to test:
# process_video('Test_set/vedio3.mp4')
process_image('Test_set/image.jpeg')
# process_image('Test_set/image_T.jpeg')






# _______or_______


# import cv2
# import torch
# import os
# import pyttsx3

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0

#     if not cap.isOpened():
#         print("❌ Error: Cannot open video.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)
#         detections = results.pandas().xyxy[0]
#         print(f"Frame {frame_count} detections:\n", detections)

#         # Draw bounding boxes and labels
#         for _, row in detections.iterrows():
#             x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             label = str(row['name'])
#             conf = float(row['confidence'])

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         cv2.imshow('Traffic Sign Detection', frame)
#         frame_count += 1

#         # Press 'q' to quit video display
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def process_image(image_path):
#     if not os.path.exists(image_path):
#         print(f"❌ Error: Image file '{image_path}' not found.")
#         return

#     img = cv2.imread(image_path)
#     results = model(img)
#     detections = results.pandas().xyxy[0]
#     print("Image detections:\n", detections)

#     # Draw results
#     for _, row in detections.iterrows():
#         x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#         label = str(row['name'])
#         conf = float(row['confidence'])

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     cv2.imshow("Image Detection", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Example usage
# # process_video('vedio2.mp4')
# process_image('image.jpeg')
