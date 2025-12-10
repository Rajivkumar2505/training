
from ultralytics import YOLO
import cv2
import os


model = YOLO("runs/detect/train4/weights/best.pt") 
# model = YOLO("RBC_G001.pt")


image_path = "AGAR_demo/AGAR_representative/higher-resolution/dark/5206.jpg"


image=cv2.imread(image_path)
results = model(image, conf = 0.3, save = False, max_det = 2000)

det = results[0].boxes

# Create a copy of the image for annotation
annotated = image.copy()

rbc_num = len(det)

# Draw the bounding boxes on the image (no labels, no coordinates)
for box in det:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the box coordinates
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the box

    # cv2.circle(annotated, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (0, 0, 255), -1)  # Draw center point

# Show the annotated image without labels or coordinates
cv2.imshow(f"RBC : {rbc_num}", cv2.resize(annotated, (annotated.shape[1]//3, annotated.shape[0]//3)))

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()








