
from pathlib import Path

import cv2
from ultralytics import YOLO

# Project settings
VIDEO_PATH = Path("videos/people.mp4")
MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.3
LINE_Y_RATIO = 0.60

# Display settings
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PLAYBACK_DELAY = 30

# Check video file
if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}. "
        "Check the filename inside the videos folder."
    )

# Load pretrained YOLO model
model = YOLO(MODEL_NAME)

# Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

entered_ids = set()
exited_ids = set()
previous_positions = {}

print("Starting Intelligent Video Analytics...")

while True:
    success, frame = cap.read()

    if not success:
        break

    height, width = frame.shape[:2]
    line_y = int(height * LINE_Y_RATIO)

    # Detect and track people
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    # Draw the virtual counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3
    )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(xyxy, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Center of the detected person
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                previous_y = previous_positions.get(track_id)

                # Detect line crossings
                if previous_y is not None:
                    crossed_down = (
                        previous_y < line_y <= center_y
                    )
                    crossed_up = (
                        previous_y > line_y >= center_y
                    )

                    if (
                        crossed_down
                        and track_id not in entered_ids
                    ):
                        entered_ids.add(track_id)

                    elif (
                        crossed_up
                        and track_id not in exited_ids
                    ):
                        exited_ids.add(track_id)

                previous_positions[track_id] = center_y

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Draw center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    6,
                    (0, 0, 255),
                    -1
                )

                # Draw larger Person ID label
                cv2.putText(
                    frame,
                    f"Person ID: {track_id}",
                    (x1, max(y1 - 15, 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

    # Draw larger Entered counter
    cv2.putText(
        frame,
        f"Entered: {len(entered_ids)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    # Draw larger Exited counter
    cv2.putText(
        frame,
        f"Exited: {len(exited_ids)}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    # Resize only the displayed frame
    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "Intelligent Video Analytics",
        display_frame
    )

    # Press Q to quit
    if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("\nVideo processing finished.")
print("Total entered:", len(entered_ids))
print("Total exited:", len(exited_ids))
