
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
MOVEMENT_THRESHOLD = 8

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

entered_ids = set()
exited_ids = set()

# Store the previous (x, y) position of each tracked ID
previous_positions = {}

print("Starting Intelligent Video Analytics...")

while True:
    success, frame = cap.read()

    if not success:
        break

    height, width = frame.shape[:2]
    line_y = int(height * LINE_Y_RATIO)

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    # Draw the counting line
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
            coordinates = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(coordinates, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Calculate the person's center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                previous_position = previous_positions.get(track_id)
                direction = "Initializing"

                if previous_position is not None:
                    previous_x, previous_y = previous_position

                    dx = center_x - previous_x
                    dy = center_y - previous_y

                    # Classify the dominant movement direction
                    if max(abs(dx), abs(dy)) < MOVEMENT_THRESHOLD:
                        direction = "Stationary"
                    elif abs(dx) >= abs(dy):
                        direction = (
                            "Moving Right" if dx > 0
                            else "Moving Left"
                        )
                    else:
                        direction = (
                            "Moving Down" if dy > 0
                            else "Moving Up"
                        )

                    # Count crossings of the horizontal line
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

                previous_positions[track_id] = (
                    center_x, center_y
                )

                # Draw bounding box and center
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Display person ID
                cv2.putText(
                    frame,
                    f"Person ID: {track_id}",
                    (x1, max(y1 - 40, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Display movement direction
                cv2.putText(
                    frame,
                    direction,
                    (x1, max(y1 - 10, 55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    # Display counters
    cv2.putText(
        frame,
        f"Entered: {len(entered_ids)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Exited: {len(exited_ids)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "Intelligent Video Analytics",
        display_frame
    )

    if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVideo processing finished.")
print("Total entered:", len(entered_ids))
print("Total exited:", len(exited_ids))

