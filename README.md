# YOLO Object Detection System

## 📌 Project Overview

This project is a Computer Vision application developed using the
YOLO (You Only Look Once) framework.

The system detects and identifies objects in:

- Images
- Video files
- Real-time webcam streams

For every detected object, the system displays a bounding box,
object class, and confidence score.

The project is developed using Python, Ultralytics YOLO, OpenCV,
and PyTorch in Visual Studio Code.

---

## 🎯 Objectives

The main objectives of this project are:

1. To understand the working of YOLO-based object detection.
2. To detect multiple objects in an image.
3. To perform real-time object detection using a webcam.
4. To detect objects frame-by-frame in video files.
5. To display bounding boxes and confidence scores.
6. To save the processed video with detected objects.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| YOLO | Object detection |
| Ultralytics | YOLO framework |
| OpenCV | Image, video and webcam processing |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| VS Code | Development environment |

---

## 📂 Project Structure

```text
YOLO_Object_Detection/
│
├── images/
│   └── test.jpg
│
├── videos/
│   └── test.mp4
│
├── models/
│
├── output/
│   └── detected_video.mp4
│
├── runs/
│   └── detect/
│       └── predict/
│
├── venv/
│
├── detect_image.py
├── detect_video.py
├── detect_webcam.py
├── object_detector.py
│
├── requirements.txt
├── README.md
└── yolo26n.pt

```

📁 Description of Files

detect_image.py

Detects objects from an input image.
The detected objects are displayed with:
- Bounding boxes
- Class names
- Confidence scores

detect_video.py

Processes a video frame-by-frame using YOLO.
The detected video is saved inside:
output/detected_video.mp4

detect_webcam.py

Uses the computer's webcam for real-time object detection.
The program continuously captures frames and passes them
to the YOLO model for detection.
Press Q to stop the webcam detection.

## yolo26n.pt
This is the pretrained YOLO model used for object detection.
The model can recognize common objects from the pretrained dataset,
such as:
- Person
- Car
- Bicycle
- Dog
- Cat
- Chair
- Laptop
- Bottle
- Bus
- Motorcycle
- and many others

## ⚙️ Installation
Step 1: Clone or download the project
Open the project folder in VS Code.
Step 2: Create a virtual environment
python -m venv venv

Step 3: Activate the virtual environment
For Windows PowerShell:
```bash
.\venv\Scripts\Activate.ps1
```

Step 4: Install dependencies
```bash
pip install -r requirements.txt
```

The main dependencies are:
ultralytics
opencv-python
numpy

## ▶️ Running the Project
Image Detection
Place an image inside:
images/test.jpg

Run:
```bash
python detect_image.py
```

Webcam Detection
Run:
```bash
python detect_webcam.py
```

The webcam will open and YOLO will detect objects in real time.
Press:
Q

to exit.
Video Detection
Place a video inside:
videos/test.mp4

Run:
```bash
python detect_video.py
```

The processed video will be saved as:
output/detected_video.mp4

Complete Application
Run:
python object_detector.py

The following menu will appear:
 YOLO OBJECT DETECTOR
1. Detect objects in image
2. Detect objects using webcam
3. Detect objects in video
4. Exit

## 🧠 How the System Works
The overall workflow is:
```text
             Input
               │
       ┌───────┼────────┐
       │       │        │
     Image   Video    Webcam
       │       │        │
       └───────┼────────┘
               ↓
          OpenCV
               ↓
          YOLO Model
               ↓
      Object Detection
               ↓
    ┌──────────┼──────────┐
    │          │          │
 Class Name  Confidence  Bounding Box
    │          │          │
    └──────────┼──────────┘
               ↓
       Display / Save

```

## 🔍 YOLO Detection
YOLO stands for:
You Only Look Once
It is a real-time object detection algorithm.
Instead of separately searching different regions of an image,
YOLO processes the image using a single neural-network-based
detection pipeline.
For each detected object, the model provides:
Object Class
Confidence Score
Bounding Box Coordinates

For example:
Person     0.94
Dog        0.87
Car        0.91

A confidence threshold is used to control which detections are
displayed.
The project currently uses:
conf=0.5
which means detections below the selected confidence threshold
are filtered out.

## 📦 Main Python Libraries
Ultralytics
Used to load and run the YOLO model.
Example:
from ultralytics import YOLOmodel = YOLO("yolo26n.pt")


OpenCV
Used for:
- Reading images
- Reading videos
- Accessing webcam
- Displaying frames
- Saving processed videos
Example:
import cv2camera = cv2.VideoCapture(0)


📊 Output
```text
The system produces detections similar to:
Person - 95%
Laptop - 89%
Chair - 82%
Bottle - 76%
```

Bounding boxes are drawn around detected objects.
For video detection, the processed video is stored in:
output/detected_video.mp4

## 🚀 Applications
This project can be extended for:
- Smart surveillance
- Traffic monitoring
- Vehicle detection
- Pedestrian detection
- Smart parking
- Industrial safety
- Crowd monitoring
- Object counting
- Robotics
- Autonomous systems

## 🔮 Future Scope
The current project uses a pretrained YOLO model.
It can be further improved by training YOLO on a custom dataset for
specific applications such as:
- Helmet detection
- Pothole detection
- Waste detection
- Fire detection
- Traffic sign detection
- Face-mask detection
- Plant disease detection
A custom-trained model can be used instead of the pretrained model
to detect domain-specific objects.


Developed as a Computer Vision project using:
Python + YOLO + OpenCV + Ultralytics
