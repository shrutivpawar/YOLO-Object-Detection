from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam started")
print("Press Q to quit")

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read frame")
        break

    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    detected_objects = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        detected_objects.append(
            f"{class_name}: {confidence:.2f}"
    )

    print(detected_objects)

    annotated_frame = result.plot()

    object_count = len(result.boxes)

    cv2.putText(
        annotated_frame,
        f"Objects Detected: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO26 Real-Time Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()

cv2.destroyAllWindows()