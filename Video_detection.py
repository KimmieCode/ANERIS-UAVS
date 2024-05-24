import cv2
from ultralytics import YOLO

# Define the path to the trained model
relative_model_path = r'C:\Machine_learning_bachelor\runs\detect\train13\weights\last.pt'

# the path to the trained model within the same directory as the source code
dir_model_path = r'C:\Machine_learning_bachelor\last.pt'

# Load the trained YOLOv8n model
model = YOLO(dir_model_path)

# Load the video
video_path = r"C:\Machine_learning_bachelor\video_samples\fish_swimming_5.mp4"
#cap = cv2.VideoCapture(video_path)

# detected fish videos get saved in the \runs\detect directory
results = model(source=video_path, show=True, conf=0.4, save=True)




