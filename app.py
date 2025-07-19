#Red Light Violation Detection System
#Created by Kunal Gandvane
#GitHub: https://github.com/webserver105
#Date: 19th July, 2025

#Features:
#- Real-time vehicle tracking
#- Traffic light state detection
#- Red Light Violation Detection
#- Violation recording with timestamps stored in logs


from flask import Flask, render_template, Response, jsonify, send_from_directory, send_file, request
import cv2
import csv
import threading
import numpy as np
from collections import deque
from datetime import datetime
import os
import tempfile
from werkzeug.utils import secure_filename
from detectors import load_models, detect_traffic_lights
from logic import check_violations
from utils import (setup_directories, initialize_csv, log_violation_to_csv,
                   save_violation_clip, draw_annotations, get_vehicle_class)
import config
from deep_sort_pytorch.utils.parser import get_config as get_deepsort_config
from deep_sort_pytorch.deep_sort import DeepSort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

processing_active = False 
processing_lock = threading.Lock()
latest_violations = []

def initialize_tracker():
    cfg = get_deepsort_config()
    config_path = os.path.join("deep_sort_pytorch", "configs", "deep_sort.yaml")
    cfg.merge_from_file(config_path)
    return DeepSort(
        cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True
    )

def generate_frames():
    global processing_active
    
    setup_directories()
    initialize_csv()
    vehicle_model, traffic_light_model = load_models()
    tracker = initialize_tracker()
    
    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.INPUT_VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    buffer_size = int((config.RECORD_SECONDS_BEFORE + config.RECORD_SECONDS_AFTER) * fps)
    frame_buffer = deque(maxlen=buffer_size)

    frame_count = 0
    traffic_light_status = "Unknown"
    violation_ids = set()

    while cap.isOpened() and processing_active:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_buffer.append(frame.copy())

        if frame_count % 10 == 0:
            traffic_light_status = detect_traffic_lights(traffic_light_model, frame)
        
        vehicle_results = vehicle_model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]

        detections = []
        for box in vehicle_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf, cls = float(box.conf[0].cpu().numpy()), int(box.cls[0].cpu().numpy())
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        
        bboxes_xywh = np.array([d[0] for d in detections]) if detections else np.empty((0, 4))
        confidences = np.array([d[1] for d in detections]) if detections else np.empty((0,))
        classes = np.array([d[2] for d in detections]) if detections else np.empty((0,))
        
        tracker.update(bboxes_xywh, confidences, classes, frame)
        active_tracks = tracker.tracker.tracks

        violation_ids, new_violators = check_violations(active_tracks, traffic_light_status, config.VIOLATION_ZONE, violation_ids)

        if new_violators and len(frame_buffer) >= buffer_size:
            for violator_track in new_violators:
                track_id = violator_track.track_id
                class_id = violator_track.cls
                vehicle_class = get_vehicle_class(int(class_id))
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                clip_filename = f"violation_id_{track_id}_{timestamp_str}.mp4"
                
                save_violation_clip(list(frame_buffer), clip_filename, fps, frame_size)
                violation_data = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Vehicle ID': str(track_id),
                    'Vehicle Class': vehicle_class,
                    'Clip Filename': clip_filename
                }
                log_violation_to_csv(track_id, vehicle_class, clip_filename)
                with processing_lock:
                    latest_violations.append(violation_data)

        draw_annotations(frame, active_tracks, traffic_light_status, config.VIOLATION_ZONE, violation_ids)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    processing_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_processing')
def start_processing():
    global processing_active
    if not processing_active:
        processing_active = True
    return jsonify(status="started")

@app.route('/stop_processing')
def stop_processing():
    global processing_active
    processing_active = False
    return jsonify(status="stopped")

@app.route('/get_violations')
def get_violations():
    violations = []
    
    try:
        if os.path.exists(config.CSV_LOG_PATH):
            with processing_lock:
                with open(config.CSV_LOG_PATH, 'r') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and 'Clip Filename' in reader.fieldnames:
                        csv_violations = [row for row in reader]
                        violations.extend(csv_violations[-50:])
    
        with processing_lock:
            violations.extend(latest_violations)
        
        unique_violations = []
        seen = set()
        for v in reversed(violations):
            if 'Clip Filename' in v and v['Clip Filename'] not in seen:
                seen.add(v['Clip Filename'])
                unique_violations.append(v)
        
        return jsonify(unique_violations[-50:])
    
    except Exception as e:
        print(f"Error getting violations: {e}")
        return jsonify([])

@app.route('/logs/clips/<filename>')
def serve_clip(filename):
    return send_file(os.path.join(config.CLIPS_DIR, filename), mimetype='video/mp4', as_attachment=False)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify(error="No video file provided"), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify(error="No selected file"), 400
    
    if video_file:
        upload_dir = os.path.join('data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(upload_dir, filename)
        video_file.save(filepath)
        
        config.INPUT_VIDEO_PATH = filepath
        return jsonify(success=True, filename=filename)
    
    return jsonify(error="Invalid file"), 400

@app.route('/use_live_camera')
def use_live_camera():
    config.INPUT_VIDEO_PATH = 0
    return jsonify(success=True)

@app.route('/update_zone', methods=['POST'])
def update_zone():
    try:
        points = request.json.get('points')
        if not points or len(points) < 3:
            return jsonify(error="At least 3 points required"), 400
            
        config.VIOLATION_ZONE = np.array(points, np.int32)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)