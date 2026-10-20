
from pathlib import Path
from collections import deque
from datetime import datetime
import csv

import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================
# 1. SETTINGS
# ==================================================

VIDEO_PATH = Path("videos/people.mp4")
MODEL_NAME = "yolo11n.pt"
LOG_PATH = Path("logs/intrusion_events.csv")

CONFIDENCE = 0.3

# Line-crossing settings
LINE_Y_RATIO = 0.60
SAFETY_MARGIN = 15

# Restricted-zone polygon (normalized coordinates)
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

# Dashboard settings
MAX_RECENT_EVENTS = 5

# ==================================================
# 2. CSV AND EVENT FUNCTIONS
# ==================================================

def format_video_timestamp(milliseconds):
    """Format a video timestamp as HH:MM:SS.mmm."""
    total_ms = max(0, int(milliseconds))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1_000
    millis = total_ms % 1_000

    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def log_event(writer, log_file, recent_events,
              person_id, event_type, video_timestamp):
    """Save an event to CSV and add it to the live dashboard."""

    recorded_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    event = {
        "recorded_at": recorded_at,
        "video_timestamp": video_timestamp,
        "person_id": person_id,
        "event_type": event_type,
    }

    # Save the event permanently in CSV
    writer.writerow([
        recorded_at,
        video_timestamp,
        person_id,
        event_type,
    ])

    log_file.flush()

    # Add the newest event to the front of the dashboard
    recent_events.appendleft(event)

    print(
        f"{event_type}: Person ID {person_id} "
        f"at video time {video_timestamp}"
    )


def draw_text(frame, text, position, color=(255, 255, 255),
              scale=0.6, thickness=2):
    """Draw readable text on the video frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ==================================================
# 3. INITIALIZATION
# ==================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Append mode preserves events from previous runs
log_file = LOG_PATH.open(
    "a",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(log_file)

# Write column headers only when the file is empty
if log_file.tell() == 0:
    csv_writer.writerow([
        "recorded_at",
        "video_timestamp",
        "person_id",
        "event_type",
    ])
    log_file.flush()

# Store only the five newest events in memory
recent_events = deque(maxlen=MAX_RECENT_EVENTS)

# Person movement history
previous_positions = {}

# Previous stable side of the counting line
last_sides = {}

# Previous restricted-zone state per tracking ID
previous_zone_states = {}

# Line-crossing totals
entered_count = 0
exited_count = 0

# Restricted-zone totals
zone_entry_events = 0
zone_exit_events = 0

# Distinct tracking IDs that entered the zone
intrusion_ids_seen = set()

model = None
cap = None

try:
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    print("Starting Intelligent Video Analytics...")
    print(f"CSV event log: {LOG_PATH}")

    # ==================================================
    # 4. VIDEO PROCESSING LOOP
    # ==================================================

    while True:
        success, frame = cap.read()

        if not success:
            break

        height, width = frame.shape[:2]
        line_y = int(height * LINE_Y_RATIO)

        video_timestamp = format_video_timestamp(
            cap.get(cv2.CAP_PROP_POS_MSEC)
        )

        # Convert normalized polygon points to pixel coordinates
        zone = np.array(
            [
                (int(x * width), int(y * height))
                for x, y in ZONE_POINTS
            ],
            dtype=np.int32,
        )

        # ----------------------------------------------
        # YOLO DETECTION + BYTETRACK TRACKING
        # ----------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE,
            verbose=False,
        )

        # ----------------------------------------------
        # DRAW RESTRICTED ZONE
        # ----------------------------------------------

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [zone],
            (0, 0, 255),
        )

        frame = cv2.addWeighted(
            overlay,
            0.22,
            frame,
            0.78,
            0,
        )

        cv2.polylines(
            frame,
            [zone],
            True,
            (0, 0, 255),
            3,
        )

        draw_text(
            frame,
            "RESTRICTED ZONE",
            (
                max(10, int(zone[0][0])),
                max(30, int(zone[0][1]) - 10),
            ),
            (0, 0, 255),
            0.7,
            2,
        )

        # ----------------------------------------------
        # DRAW COUNTING LINE
        # ----------------------------------------------

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

        # IDs inside the zone in this frame
        active_zone_ids = set()

        # ----------------------------------------------
        # PROCESS DETECTED PEOPLE
        # ----------------------------------------------

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()

                for box, track_id in zip(
                    coordinates, track_ids
                ):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Approximate foot point
                    foot_x = center_x
                    foot_y = y2

                    # ----------------------------------
                    # MOVEMENT DIRECTION
                    # ----------------------------------

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
                                "Moving Right"
                                if dx > 0
                                else "Moving Left"
                            )

                        else:
                            direction = (
                                "Moving Down"
                                if dy > 0
                                else "Moving Up"
                            )

                    previous_positions[track_id] = (
                        center_x,
                        center_y,
                    )

                    # ----------------------------------
                    # RESTRICTED-ZONE EVENT DETECTION
                    # ----------------------------------

                    inside_zone = cv2.pointPolygonTest(
                        zone,
                        (float(foot_x), float(foot_y)),
                        False,
                    ) >= 0

                    was_inside = previous_zone_states.get(
                        track_id, False
                    )

                    # Outside -> inside: ENTRY
                    if inside_zone and not was_inside:
                        zone_entry_events += 1
                        intrusion_ids_seen.add(track_id)

                        log_event(
                            csv_writer,
                            log_file,
                            recent_events,
                            track_id,
                            "ENTRY",
                            video_timestamp,
                        )

                    # Inside -> outside: EXIT
                    elif was_inside and not inside_zone:
                        zone_exit_events += 1

                        log_event(
                            csv_writer,
                            log_file,
                            recent_events,
                            track_id,
                            "EXIT",
                            video_timestamp,
                        )

                    previous_zone_states[track_id] = inside_zone

                    if inside_zone:
                        active_zone_ids.add(track_id)

                    # ----------------------------------
                    # LINE-CROSSING COUNTS
                    # ----------------------------------

                    if foot_y < line_y - SAFETY_MARGIN:
                        current_side = -1
                    elif foot_y > line_y + SAFETY_MARGIN:
                        current_side = 1
                    else:
                        current_side = None

                    previous_side = last_sides.get(track_id)

                    if current_side is not None:
                        if (
                            previous_side == -1
                            and current_side == 1
                        ):
                            entered_count += 1

                        elif (
                            previous_side == 1
                            and current_side == -1
                        ):
                            exited_count += 1

                        last_sides[track_id] = current_side

                    # ----------------------------------
                    # DRAW PERSON ANNOTATIONS
                    # ----------------------------------

                    box_color = (
                        (0, 0, 255)
                        if inside_zone
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

                    draw_text(
                        frame,
                        f"Person ID: {track_id}",
                        (x1, max(y1 - 40, 30)),
                        box_color,
                        0.7,
                        2,
                    )

                    draw_text(
                        frame,
                        direction,
                        (x1, max(y1 - 12, 55)),
                        (255, 255, 0),
                        0.65,
                        2,
                    )

                    if inside_zone:
                        draw_text(
                            frame,
                            "INTRUSION ALERT!",
                            (x1, min(y2 + 25, height - 10)),
                            (0, 0, 255),
                            0.7,
                            2,
                        )

        # ==================================================
        # 5. LIVE EVENT DASHBOARD
        # ==================================================

        # Draw dashboard over the right side of the frame
        panel_width = min(350, max(220, width - 20))
        panel_left = max(10, width - panel_width - 10)
        panel_top = 10
        panel_height = 300

        cv2.rectangle(
            frame,
            (panel_left, panel_top),
            (
                min(width - 1, panel_left + panel_width),
                min(height - 1, panel_top + panel_height),
            ),
            (30, 30, 30),
            -1,
        )

        text_x = panel_left + 12

        draw_text(
            frame,
            "INTRUSION DASHBOARD",
            (text_x, 38),
            (255, 255, 255),
            0.58,
            2,
        )

        draw_text(
            frame,
            f"Zone entries: {zone_entry_events}",
            (text_x, 75),
            (0, 255, 0),
            0.58,
            2,
        )

        draw_text(
            frame,
            f"Zone exits: {zone_exit_events}",
            (text_x, 108),
            (0, 165, 255),
            0.58,
            2,
        )

        draw_text(
            frame,
            f"People inside: {len(active_zone_ids)}",
            (text_x, 141),
            (0, 0, 255),
            0.58,
            2,
        )

        draw_text(
            frame,
            f"Unique IDs: {len(intrusion_ids_seen)}",
            (text_x, 174),
            (255, 255, 255),
            0.58,
            2,
        )

        draw_text(
            frame,
            "RECENT EVENTS",
            (text_x, 208),
            (0, 255, 255),
            0.55,
            2,
        )

        for index, event in enumerate(recent_events):
            y = 235 + index * 18

            if y >= min(height - 5, panel_top + panel_height):
                break

            event_text = (
                f"{event['event_type']} "
                f"ID:{event['person_id']} "
                f"{event['video_timestamp']}"
            )

            # Keep text within the panel on typical video sizes
            max_chars = max(18, (panel_width - 20) // 8)
            event_text = event_text[:max_chars]

            draw_text(
                frame,
                event_text,
                (text_x, y),
                (255, 255, 255),
                0.42,
                1,
            )

        # ----------------------------------------------
        # LEFT-SIDE COUNTERS
        # ----------------------------------------------

        counters = [
            (f"Entered: {entered_count}", 40, (0, 255, 0)),
            (f"Exited: {exited_count}", 80, (0, 0, 255)),
        ]

        for label, y, color in counters:
            draw_text(
                frame,
                label,
                (20, y),
                color,
                0.8,
                2,
            )

        # ----------------------------------------------
        # DISPLAY VIDEO
        # ----------------------------------------------

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
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
    log_file.close()

# ==================================================
# 6. FINAL REPORT
# ==================================================

print("\nVideo processing finished.")
print("Total line entries:", entered_count)
print("Total line exits:", exited_count)
print("Restricted-zone entries:", zone_entry_events)
print("Restricted-zone exits:", zone_exit_events)
print("Unique people IDs alerted:", len(intrusion_ids_seen))
print(f"Event log saved to: {LOG_PATH}")
