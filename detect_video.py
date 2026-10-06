from ultralytics import YOLO
import cv2
import os

model = YOLO("yolo26n.pt")

video_path = "videos/test.mp4"

if not os.path.exists(video_path):
    print("Video file not found.")
    exit()

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

os.makedirs("output", exist_ok=True)

output_path = "output/detected_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Processing video...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    writer.write(annotated_frame)

    cv2.imshow(
        "YOLO Video Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
cv2.destroyAllWindows()

print("Detection completed.")

print(
    f"Output saved to: {output_path}"
)