# Intelligent Video Analytics

An AI-powered video analytics project built using Python, YOLO, ByteTrack, and OpenCV.

## 🚀 Overview

This project detects and tracks people in videos and counts when tracked people cross a virtual line.

It displays bounding boxes, person IDs, entry counts, and exit counts on the video.

## 🧠 Technologies

- Python
- YOLO
- Ultralytics
- ByteTrack
- OpenCV

## 🔄 Pipeline

Video Input
↓
YOLO Person Detection
↓
ByteTrack Object Tracking
↓
Virtual Line Crossing
↓
Entry and Exit Counting
↓
Annotated Video Output

## 📁 Project Structure

```text
intelligent-videoanalytics/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── videos/
```

## ⚙️ Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Usage

Place a video named `people.mp4` inside the `videos` folder.

Run:

```bash
python main.py
```

The first run downloads the pretrained YOLO model. The program displays the video with bounding boxes, tracking IDs, and entry/exit counters.

## 🧠 What I Learned

- Computer Vision
- YOLO object detection
- Multi-object tracking with ByteTrack
- Video processing with OpenCV
- Line-crossing detection
- People counting
- Pretrained AI models

## 🔮 Future Improvements

- Vehicle counting
- Movement direction analysis
- Restricted-zone detection
- Real-time alerts
- FastAPI integration
- Advanced intelligent video analytics