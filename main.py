
from pathlib import Path
import csv
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- SETTINGS ----------------

VIDEO_PATH = Path("videos/people.mp4")
MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.3

# Line-crossing settings
LINE_Y_RATIO = 0.60
SAFETY_MARGIN = 15

# Restricted zone: normalized coordinates (0.0 to 1.0)
# Adjust these points to match your video.
ZONE_POINTS = [
    (0.58, 0.20),
    (0.92, 0.20),
    (0.95, 0.85),
    (0.58, 0.85),
]

# Display settings
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PLAYBACK_DELAY = 30
MOVEMENT_THRESHOLD = 8

# CSV event log
LOG_PATH = Path("logs/intrusion_events.csv")

# ---------------- HELPER FUNCTIONS ----------------

def format_video_timestamp(milliseconds):
    """Convert video position from milliseconds to HH:MM:SS.mmm."""
    total_ms = max(0, int(milliseconds))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1_000
    millis = total_ms % 1_000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def log_event(writer, person_id, event_type, video_timestamp):
    """Write one intrusion event to the CSV log."""
    recorded_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    writer.writerow([
        recorded_at,
        video_timestamp,
        person_id,
        event_type,
    ])

    print(
        f"{event_type}: Person ID {person_id} "
        f"at video time {video_timestamp}"
    )


# ---------------- INITIALIZATION ----------------

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Append events so previous runs remain in the log.
log_file = LOG_PATH.open(
    "a",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(log_file)

# Write the header only for a new or empty file.
if log_file.tell() == 0:
    csv_writer.writerow([
        "recorded_at",
        "video_timestamp",
        "person_id",
        "event_type",
    ])
    log_file.flush()

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    log_file.close()
    raise RuntimeError("Could not open the video.")

# Previous person center positions
previous_positions = {}

# Previous stable sides of the counting line
last_sides = {}

# Previous restricted-zone state for each tracking ID
previous_zone_states = {}

# Counts for the line
entered_count = 0
exited_count = 0

# Counts for restricted-zone transitions
zone_entry_events = 0
zone_exit_events = 0

# Unique tracking IDs that entered the restricted zone
intrusion_ids_seen = set()

print("Starting Intelligent Video Analytics...")
print(f"CSV event log: {LOG_PATH}")

# ---------------- VIDEO PROCESSING ----------------

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        height, width = frame.shape[:2]
        line_y = int(height * LINE_Y_RATIO)

        # Get the video's current playback timestamp
        video_timestamp = format_video_timestamp(
            cap.get(cv2.CAP_PROP_POS_MSEC)
        )

        # Convert normalized polygon coordinates to pixels
        zone = np.array(
            [
                (int(x * width), int(y * height))
                for x, y in ZONE_POINTS
            ],
            dtype=np.int32,
        )

        # -------- DETECTION AND TRACKING --------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE,
            verbose=False,
        )

        # -------- DRAW RESTRICTED ZONE --------

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [zone],
            (0, 0, 255),
        )

        frame = cv2.addWeighted(
            overlay, 0.22,
            frame, 0.78,
            0,
        )

        cv2.polylines(
            frame,
            [zone],
            True,
            (0, 0, 255),
            3,
        )

        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            (
                max(10, int(zone[0][0])),
                max(30, int(zone[0][1]) - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # -------- DRAW COUNTING LINE --------

        cv2.line(
            frame,
            (0, line_y),
            (width, line_y),
            (0, 255, 255),
            3,
        )

        cv2.line(
            frame,
            (0, line_y - SAFETY_MARGIN),
            (width, line_y - SAFETY_MARGIN),
            (180, 180, 180),
            1,
        )

        cv2.line(
            frame,
            (0, line_y + SAFETY_MARGIN),
            (width, line_y + SAFETY_MARGIN),
            (180, 180, 180),
            1,
        )

        # People currently inside the zone
        active_zone_ids = set()

        # -------- PROCESS TRACKED PEOPLE --------

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()

                for box, track_id in zip(coordinates, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Approximate foot position
                    foot_x = center_x
                    foot_y = y2

                    # -------- MOVEMENT DIRECTION --------

                    previous_position = previous_positions.get(
                        track_id
                    )

                    direction = "Initializing"

                    if previous_position is not None:
                        previous_x, previous_y = previous_position

                        dx = center_x - previous_x
                        dy = center_y - previous_y

                        if (
                            max(abs(dx), abs(dy))
                            < MOVEMENT_THRESHOLD
                        ):
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

                    previous_positions[track_id] = (
                        center_x,
                        center_y,
                    )

                    # -------- RESTRICTED-ZONE DETECTION --------

                    inside_zone = cv2.pointPolygonTest(
                        zone,
                        (float(foot_x), float(foot_y)),
                        False,
                    ) >= 0

                    was_inside = previous_zone_states.get(
                        track_id, False
                    )

                    # ENTRY EVENT
                    if inside_zone and not was_inside:
                        zone_entry_events += 1
                        intrusion_ids_seen.add(track_id)

                        log_event(
                            csv_writer,
                            track_id,
                            "ENTRY",
                            video_timestamp,
                        )

                        # Persist each event immediately
                        log_file.flush()

                    # EXIT EVENT
                    elif was_inside and not inside_zone:
                        zone_exit_events += 1

                        log_event(
                            csv_writer,
                            track_id,
                            "EXIT",
                            video_timestamp,
                        )

                        log_file.flush()

                    # Remember the current zone state
                    previous_zone_states[track_id] = inside_zone

                    if inside_zone:
                        active_zone_ids.add(track_id)

                    # -------- LINE-CROSSING COUNTS --------

                    if foot_y < line_y - SAFETY_MARGIN:
                        current_side = -1

                    elif foot_y > line_y + SAFETY_MARGIN:
                        current_side = 1

                    else:
                        current_side = None

                    previous_side = last_sides.get(track_id)

                    if current_side is not None:
                        if previous_side == -1 and current_side == 1:
                            entered_count += 1

                        elif previous_side == 1 and current_side == -1:
                            exited_count += 1

                        last_sides[track_id] = current_side

                    # -------- DRAW PERSON ANNOTATIONS --------

                    box_color = (
                        (0, 0, 255) if inside_zone
                        else (0, 255, 0)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        3,
                    )

                    cv2.circle(
                        frame,
                        (foot_x, foot_y),
                        5,
                        (255, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        f"Person ID: {track_id}",
                        (x1, max(y1 - 40, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        direction,
                        (x1, max(y1 - 12, 55)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    if inside_zone:
                        cv2.putText(
                            frame,
                            "INTRUSION ALERT!",
                            (x1, min(y2 + 25, height - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

        # -------- DISPLAY COUNTERS --------

        counters = [
            (f"Entered: {entered_count}", 40, (0, 255, 0)),
            (f"Exited: {exited_count}", 80, (0, 0, 255)),
            (
                f"People in zone: {len(active_zone_ids)}",
                120,
                (0, 0, 255),
            ),
            (
                f"Unique IDs alerted: {len(intrusion_ids_seen)}",
                160,
                (0, 0, 255),
            ),
            (
                f"Zone Entries: {zone_entry_events}",
                200,
                (0, 0, 255),
            ),
            (
                f"Zone Exits: {zone_exit_events}",
                240,
                (0, 0, 255),
            ),
        ]

        for label, y, color in counters:
            cv2.putText(
                frame,
                label,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # -------- DISPLAY VIDEO --------

        display_frame = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        )

        cv2.imshow(
            "Intelligent Video Analytics",
            display_frame,
        )

        if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

# ---------------- FINAL REPORT ----------------

print("\nVideo processing finished.")
print("Total line entries:", entered_count)
print("Total line exits:", exited_count)
print("Restricted-zone entries:", zone_entry_events)
print("Restricted-zone exits:", zone_exit_events)
print("Unique people IDs alerted:", len(intrusion_ids_seen))
print(f"Event log saved to: {LOG_PATH}")

