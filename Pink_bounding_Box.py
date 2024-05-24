import cv2
from ultralytics import YOLO

# Define the path to the trained model
model_path = r'C:\Machine_learning_bachelor\runs\detect\train13\weights\last.pt'

# Load the trained YOLOv8 model
model = YOLO(model_path)

# Predict with the model
source_im = r"C:\Machine_learning_bachelor\image_samples\im_sample_8.png"
results = model(source_im, conf=0.4)

# Load the image using OpenCV
image = cv2.imread(source_im)

# Define the color for the bounding boxes (e.g., blue in BGR format)
color = (162, 133, 255)  # pink color
text_color = (255, 255, 255)  # White color for text

# Iterate over the results
for result in results:
    # Each result may contain multiple detections
    for bbox in result.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # Extract bounding box coordinates
        confidence = bbox.conf.item()  # Extract confidence score
        class_id = int(bbox.cls)  # Extract class id
        class_name = model.names[class_id]  # Get class name from the model

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare the label with class name and confidence
        label = f'{class_name}: {confidence:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y1 = max(y1, label_size[1] + 10)
        cv2.rectangle(image, (x1, label_y1 - label_size[1] - 10), (x1 + label_size[0], label_y1 + 5), color, -1)
        cv2.putText(image, label, (x1, label_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

# Optionally, save the image with bounding boxes
cv2.imwrite(r"C:\Machine_learning_bachelor\image_samples\pink_bounding_boxes\im_sample_7_with_bboxes.jpg", image)

# Display the image with bounding boxes
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

