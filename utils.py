import cv2
import numpy as np
import csv
import os
from datetime import datetime
import config

def setup_directories():
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.CLIPS_DIR, exist_ok=True)

def initialize_csv():
    if not os.path.exists(config.CSV_LOG_PATH):
        with open(config.CSV_LOG_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Timestamp', 'Vehicle ID', 'Vehicle Class', 'Clip Filename'])
            writer.writeheader()

def log_violation_to_csv(track_id, vehicle_class, clip_filename):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(config.CSV_LOG_PATH)
    
    with open(config.CSV_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Vehicle ID', 'Vehicle Class', 'Clip Filename'])
        writer.writerow([timestamp, track_id, vehicle_class, clip_filename])

def save_violation_clip(frame_buffer, clip_filename, fps, frame_size):
    clip_path = os.path.join(config.CLIPS_DIR, clip_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
    
    print(f"💾 Saving violation clip: {clip_path}...")
    for frame in frame_buffer:
        writer.write(frame)
    writer.release()
    print(f"✅ Clip saved successfully.")

def get_vehicle_class(class_id):
    try:
        return config.VEHICLE_CLASS_NAMES.get(class_id, 'Unknown')
    except:
        return 'Unknown'

def draw_zone(frame, zone_polygon):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_polygon], (0, 255, 255))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [zone_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

def draw_annotations(frame, tracked_vehicles, traffic_light_status, zone_polygon, violation_ids):
    draw_zone(frame, zone_polygon)
    
    light_color = (0, 255, 0) if traffic_light_status not in config.RED_LIGHT_CLASSES else (0, 0, 255)
    cv2.putText(frame, f"Light: {traffic_light_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)

    for track in tracked_vehicles:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_id = track.track_id
        class_id = int(track.cls)
        vehicle_class = get_vehicle_class(class_id)
        
        color = (0, 0, 255) if track_id in violation_ids else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{vehicle_class} ID:{track_id}"
        
        if track_id in violation_ids:
            cv2.putText(frame, "VIOLATION", (x1, y1 - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def is_center_in_zone(box_center, zone_polygon):
    return cv2.pointPolygonTest(zone_polygon, box_center, False) >= 0