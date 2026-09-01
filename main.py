import cv2
import numpy as np
import ultralytics
import shutil
import os
from collections import Counter, deque
from smart_road.tracker import YOLODetector
from smart_road.videoreader import VideoReader


SPEED_LIMIT = 60

# 1. Calibrated Y-axis lines covering the visible lower road section
lines_y = {
    1: 340,  # Top measurement line
    2: 400,
    3: 470,
    4: 550,
    5: 640   # Bottom measurement line
}

line_colors = [
    (255, 0, 128),  # Cyber Pink
    (0, 242, 255),  # Electric Blue
    (255, 0, 128),
    (0, 242, 255),
    (0, 255, 170)   # Neon Cyan
]

# 2. Distance mapping due to perspective projection
pair_distances = {
    (1, 2): 8.0,
    (2, 3): 6.0,
    (3, 4): 4.0,
    (4, 5): 3.2
}

vehicle_speeds = {}
vehicle_times = {}

# Buffers to stabilize dynamic metrics over time
STATE_BUFFER_SIZE = 30
condition_history = deque(maxlen=STATE_BUFFER_SIZE)
count_history = deque(maxlen=STATE_BUFFER_SIZE)
speed_history = deque(maxlen=STATE_BUFFER_SIZE)


def vehicle_crossing_line(track_id, vehicle_y, lines, vehicle_times, current_frame, fps):

    if track_id not in vehicle_times:
        first_line_v = list(lines.values())[0]
        check_greater = vehicle_y < first_line_v
        vehicle_times[track_id] = {'use_greater': check_greater}

    line_keys = list(lines.keys())
    current_time_sec = current_frame / fps

    for k, v in lines.items():
        if k in vehicle_times[track_id]:
            continue

        idx = line_keys.index(k)

        if vehicle_times[track_id]['use_greater']:
            # vehicle moving downward
            if vehicle_y >= v:
                vehicle_times[track_id][k] = current_time_sec
                if idx > 0:
                    prev_k = line_keys[idx - 1]
                    if prev_k in vehicle_times[track_id]:
                        elapsed_time = vehicle_times[track_id][k] - vehicle_times[track_id][prev_k]
                        if elapsed_time > 0:
                            dist = pair_distances.get((prev_k, k), 10.0)
                            speed_ms = dist / elapsed_time
                            vehicle_speeds[track_id] = int(speed_ms * 3.6)
        else:
            # vehicle moving upward
            if vehicle_y <= v:
                vehicle_times[track_id][k] = current_time_sec
                if idx < len(line_keys) - 1:   #-1 because the last line willnot have a line after it
                    next_k = line_keys[idx + 1]
                    if next_k in vehicle_times[track_id]:
                        elapsed_time = vehicle_times[track_id][k] - vehicle_times[track_id][next_k]
                        if elapsed_time > 0:
                            dist = pair_distances.get((k, next_k), 10.0)
                            speed_ms = dist / elapsed_time
                            vehicle_speeds[track_id] = int(speed_ms * 3.6)


def analyze_traffic(vehicles):

    color_map = {
        "Nominal": (255, 255, 0),      
        "Optimal Flow": (0, 255, 128),  
        "Moderate Load": (0, 215, 255), 
        "Heavy Congestion": (0, 128, 255), 
        "CRITICAL GRIDLOCK": (50, 0, 255)  
    }

    speeds = [v["speed"] for v in vehicles if "speed" in v and v["speed"] is not None and v["speed"] > 0]
    raw_avg_speed = sum(speeds) / len(speeds) if speeds else 0
    raw_count = len(vehicles)

    if raw_count <= 3 and raw_avg_speed >= 80:
        raw_condition = "Nominal"
    elif raw_count <= 8 and raw_avg_speed >= 60:
        raw_condition = "Optimal Flow"
    elif raw_count <= 15 and raw_avg_speed >= 40:
        raw_condition = "Moderate Load"
    elif raw_count <= 30 and raw_avg_speed >= 20:
        raw_condition = "Heavy Congestion"
    else:
        raw_condition = "CRITICAL GRIDLOCK"

    condition_history.append(raw_condition)
    count_history.append(raw_count)
    speed_history.append(raw_avg_speed)

    steady_condition = Counter(condition_history).most_common(1)[0][0]
    steady_count = int(round(sum(count_history) / len(count_history)))
    steady_speed = sum(speed_history) / len(speed_history)

    return {
        "condition": steady_condition,
        "count": steady_count,
        "avg_speed": steady_speed,
        "color": color_map[steady_condition]
    }


def draw_futuristic_card(frame, x, y, w, h, border_color, title, items):
    overlay = frame.copy()
    
    # Chamfered corner polygon
    cut = 15
    pts = np.array([
        [x + cut, y], [x + w, y], 
        [x + w, y + h - cut], [x + w - cut, y + h], 
        [x, y + h], [x, y + cut]
    ], np.int32)

    # Translucent dark base
    cv2.fillPoly(overlay, [pts], (12, 16, 22))
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Glowing outer border
    cv2.polylines(frame, [pts], True, border_color, 1, lineType=cv2.LINE_AA)
    
    # Futuristic HUD Corner accents
    acc = 8
    cv2.line(frame, (x, y + cut), (x, y + cut + acc), (255, 255, 255), 2)
    cv2.line(frame, (x + cut, y), (x + cut + acc, y), (255, 255, 255), 2)
    cv2.line(frame, (x + w - cut, y + h), (x + w - cut - acc, y + h), (255, 255, 255), 2)

    # Title header
    cv2.putText(frame, title, (x + 15, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (x + 15, y + 33), (x + w - 15, y + 33), (40, 60, 80), 1)

    # Render data lines
    curr_y = y + 55
    for text, color, font_scale in items:
        cv2.putText(frame, text, (x + 15, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        curr_y += 26


def draw_speed_warning_banner(frame, speeder_count):
    """Renders a top-center warning banner when speed violations occur."""
    h, w, _ = frame.shape
    bw, bh = 360, 45
    bx, by = (w - bw) // 2, 20

    overlay = frame.copy()
    pts = np.array([
        [bx + 10, by], [bx + bw - 10, by],
        [bx + bw, by + bh], [bx, by + bh]
    ], np.int32)

    # Translucent red background callout
    cv2.fillPoly(overlay, [pts], (15, 15, 120))
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Glowing border accent
    cv2.polylines(frame, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)

    msg = f"SPEED ALERT: {speeder_count} VEHICLE(S) OVER {SPEED_LIMIT} KM/H"
    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    tx = bx + (bw - text_size[0]) // 2
    cv2.putText(frame, msg, (tx, by + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_targeting_reticle(frame, box, color, label, speed_text, conf=None):
    x1, y1, x2, y2 = box
    l = min(15, (x2 - x1) // 3, (y2 - y1) // 3)  # Reticle bracket length

    # 4 Corner brackets
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, 2, cv2.LINE_AA)

    cv2.line(frame, (x2, y1), (x2 - l, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, 2, cv2.LINE_AA)

    cv2.line(frame, (x1, y2), (x1 + l, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, 2, cv2.LINE_AA)

    cv2.line(frame, (x2, y2), (x2 - l, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - l), color, 2, cv2.LINE_AA)

    # Center target dot
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 2, color, -1)

    # Append confidence value if provided
    if conf is not None:
        full_label = f"{label} {conf:.2f}"
    else:
        full_label = label

    # Calculate dynamic tag width based on text length
    text_width = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    bg_width = max(140, text_width + 12)

    # Cyber HUD Label tag above bounding box
    tag_bg = np.array([
        [x1, y1 - 20], 
        [x1 + bg_width, y1 - 20], 
        [x1 + bg_width - 10, y1], 
        [x1, y1]
    ], np.int32)

    cv2.fillPoly(frame, [tag_bg], (10, 15, 20))
    cv2.polylines(frame, [tag_bg], True, color, 1, cv2.LINE_AA)
    
    cv2.putText(frame, full_label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Speed readout under targeting frame
    cv2.putText(frame, speed_text, (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


video_path = "/home/ahmed/speed estimation/Road traffic video for object recognition(720P_HD).mp4"
weights_path = "./best_reconstructed.pt"
output_path = "output.mp4"


class BoxSmoother:
    def __init__(self, alpha=0.75):
        self.alpha = alpha
        self.boxes = {}

    def update(self, track_id, box):
        box = np.array(box, dtype=np.float32)

        if track_id not in self.boxes:
            self.boxes[track_id] = box
        else:
            self.boxes[track_id] = self.alpha * box + (1 - self.alpha) * self.boxes[track_id]

        return self.boxes[track_id].astype(int)

    def remove_old_tracks(self, active_ids):
        active_ids = set(active_ids)
        self.boxes = {track_id: box for track_id, box in self.boxes.items() if track_id in active_ids}


# Configure custom ByteTrack parameters
default_tracker = os.path.join(
    os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml"
)
custom_tracker = "custom_bytetrack.yaml"

shutil.copy(default_tracker, custom_tracker)

with open(custom_tracker, "r") as f:
    content = f.read()

content = content.replace("track_buffer: 30", "track_buffer: 120")
content = content.replace("track_low_thresh: 0.1", "track_low_thresh: 0.05")

with open(custom_tracker, "w") as f:
    f.write(content)


# Futuristic Neon palette for vehicle classes
class_colors = {
    0: (255, 255, 0),  
    1: (255, 0, 128),  
    2: (0, 255, 128),   
    3: (0, 140, 255),   
}


def get_color(class_id):
    return class_colors.get(class_id, (200, 200, 200))


video_reader = VideoReader(source_path=video_path)

detector = YOLODetector(
    weights_path=weights_path,
    conf_threshold=0.2,
    iou_threshold=0.45,
    tracker_path=custom_tracker
)

smoother = BoxSmoother(alpha=0.75)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path, fourcc, video_reader.fps,
    (video_reader.width, video_reader.height)
)

FRAME_INTERVAL = 3
frame_index = 0
last_vehicles = []

show_lines = False        # Press 'i' to toggle line rendering
show_report_box = True    # Press 'r' to toggle cyber HUD cards

cv2.namedWindow("Speed Estimation", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = video_reader.get_frame()
        if frame is None:
            break

        if frame_index % FRAME_INTERVAL == 0:
            vehicles = detector.track_and_detect(frame=frame, img_size=640)
            last_vehicles = vehicles

            for v in vehicles:
                track_id = v["id"]
                smooth_box = smoother.update(track_id, v["bbox"])
                vehicle_y = smooth_box[3]

                vehicle_crossing_line(
                    track_id=track_id,
                    vehicle_y=vehicle_y,
                    lines=lines_y,
                    vehicle_times=vehicle_times,
                    current_frame=frame_index,
                    fps=video_reader.fps
                )
        else:
            vehicles = last_vehicles

        # Render grid/line overlays toggled with 'i'
        if show_lines:
            for line_id, y_pos in lines_y.items():
                color = line_colors[(line_id - 1) % len(line_colors)]
                cv2.line(frame, (0, y_pos), (video_reader.width, y_pos), color, 1, cv2.LINE_AA)

        class_counts = {}
        active_vehicles_data = []
        speeder_count = 0

        for v in vehicles:
            track_id = v["id"]
            smooth_box = smoother.update(track_id, v["bbox"])
            x1, y1, x2, y2 = map(int, smooth_box)

            conf = v["confidence"]
            class_id = v["class_id"]
            class_name = detector.class_names.get(class_id, str(class_id))
            
            current_speed = vehicle_speeds.get(track_id, 0)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            active_vehicles_data.append({"id": track_id, "speed": current_speed})

            # Check speeding state
            if current_speed > SPEED_LIMIT:
                speeder_count += 1
                color = (0, 0, 255)  # Bright Red for speeding vehicles
                speed_text = f"VEL: {current_speed} km/h [EXCEEDED]"
                label = f"VIOLATION #{track_id}"
            else:
                color = get_color(class_id)
                speed_text = f"VEL: {current_speed} km/h"
                label = f"{class_name.upper()} #{track_id}"

            # Draw targeting reticle and confidence score
            draw_targeting_reticle(frame, (x1, y1, x2, y2), color, label, speed_text, conf=conf)

        smoother.remove_old_tracks([v["id"] for v in vehicles])

        traffic_report = analyze_traffic(active_vehicles_data)

        # Trigger top alert banner when speeding occurs
        if speeder_count > 0:
            draw_speed_warning_banner(frame, speeder_count)

        # Draw Futuristic Cyber HUD when pressing 'r'
        if show_report_box:
            # 1. Left Telemetry Card
            left_items = [
                (f"STATUS: {traffic_report['condition']}", traffic_report["color"], 0.6),
                (f"UNITS TRACKED : {traffic_report['count']}", (220, 220, 220), 0.5),
                (f"AVG VELOCITY  : {traffic_report['avg_speed']:.1f} km/h", (0, 242, 255), 0.5),
                (f"VIOLATIONS    : {speeder_count}", (0, 0, 255) if speeder_count > 0 else (120, 120, 120), 0.5)
            ]
            draw_futuristic_card(frame, 20, 20, 320, 160, traffic_report["color"], "TRAFFIC MATRIX", left_items)

            # 2. Right Telemetry Card
            right_items = []
            if class_counts:
                for cname, count in class_counts.items():
                    right_items.append((f"{cname.upper()} : {count}", (220, 220, 220), 0.5))
            else:
                right_items.append(("NO TARGETS DETECTED", (120, 120, 120), 0.5))

            draw_futuristic_card(
                frame, 
                video_reader.width - 260, 
                20, 
                240, 
                max(120, 50 + len(right_items) * 26), 
                (0, 242, 255), 
                "OBJECT CLASSIFICATION", 
                right_items
            )

        cv2.imshow("Speed Estimation", frame)
        writer.write(frame)
        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('i'):
            show_lines = not show_lines            # Toggle laser lines
        elif key == ord('r'):
            show_report_box = not show_report_box  # Toggle Cyber HUD cards
        elif key == ord('q'):
            break

finally:
    video_reader.release()
    writer.release()
    cv2.destroyAllWindows()

print("Done!")
print(f"Output saved to: {output_path}")
