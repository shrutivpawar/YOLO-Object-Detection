from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo26n.pt")

image_path = "images/test.jpg"

if not os.path.exists(image_path):
    print(f"Error: {image_path} does not exist.")
    exit()

results = model.predict(
    source=image_path,
    conf=0.5,
    save=True
)

for result in results:

    annotated_frame = result.plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"Object: {class_name}, "
                f"Confidence: {confidence:.2f}"
            )

cv2.waitKey(0)
cv2.destroyAllWindows()