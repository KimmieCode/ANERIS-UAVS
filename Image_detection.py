
import os
import cv2
from ultralytics import YOLO


# Define the path to the trained model
#model_path = r'C:\Machine_learning_bachelor\runs\detect\train13\weights\last.pt'

model_path = r"C:\Machine_learning_bachelor\runs\detect\train13\weights\last.pt"

# Load the trained YOLOv8 model
model = YOLO(r'C:\Machine_learning_bachelor\last.pt')

# Predict with the model
source_im = r"C:\Machine_learning_bachelor\image_samples\im_sample_1.jpg"
results = model(source_im, show=True, conf=0.4, save=True)
cv2.waitKey(0)

#results = model("C:\Machine_learning_bachelor\image_samples\im_sample_7.jpg", show=True, conf=0.4, save=True)

