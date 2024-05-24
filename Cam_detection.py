import cv2
from ultralytics import YOLO

# Load a pre-trained model
model_path = r'C:\Machine_learning_bachelor\last.pt'
model = YOLO(model_path)

# Open the webcam (usually 0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with YOLOv8 model
    results = model(frame)

    # Display the results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
