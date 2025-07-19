from ultralytics import YOLO
import config

def load_models():
    print("Loading AI models...")
    vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
    traffic_light_model = YOLO(config.TRAFFIC_LIGHT_MODEL_PATH)
    print("✅ Models loaded successfully.")
    return vehicle_model, traffic_light_model

def detect_traffic_lights(model, frame):
    results = model(frame, conf=0.4, verbose=False)[0]
    if len(results.boxes) > 0:
        class_id = int(results.boxes[0].cls)
        return config.TRAFFIC_LIGHT_CLASSES.get(class_id, "Unknown")
    return "Unknown"