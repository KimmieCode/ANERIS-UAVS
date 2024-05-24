import os
import cv2
from ultralytics import YOLO

# Define the path to the input video
video_path = r'C:\Machine_learning_bachelor\fish_swimming_2.mp4'

# Define the path for the output video
video_path_out = r'C:\Machine_learning_bachelor\fish_swimming_2_out.mp4'

# Open the input video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get the frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame of the video.")
    cap.release()
    exit()

H, W, _ = frame.shape

# Create a VideoWriter object for the output video
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the path to the trained model
model_path = r'C:\Machine_learning_bachelor\runs\detect\train8\weights\last.pt'

# Load the trained YOLOv8 model
model = YOLO(model_path)

# Define the detection threshold
threshold = 0.5

# Process each frame of the input video
while ret:
    # Detect objects in the frame
    results = model(frame)

    # Check if any detections were made
    detections_made = False

    # Draw bounding boxes and labels on the frame
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            detections_made = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # If no detections were made, print a message and stop the code
    if not detections_made:
        print("No detections were made. Stopping the video processing.")
        break

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()
