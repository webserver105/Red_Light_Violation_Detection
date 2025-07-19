import numpy as np
import os

LOGS_DIR = 'logs'
CLIPS_DIR = os.path.join(LOGS_DIR, 'clips')
CSV_LOG_PATH = os.path.join(LOGS_DIR, 'violations.csv')

VEHICLE_MODEL_PATH = 'vehicle.pt'
TRAFFIC_LIGHT_MODEL_PATH = 'light.pt'

INPUT_VIDEO_PATH = 'data/demo3.mp4'
OUTPUT_VIDEO_PATH = 'output/demo3_result.mp4'
CONFIDENCE_THRESHOLD = 0.3

TRAFFIC_LIGHT_CLASSES = {
    0: 'Red',
    1: 'RedLeft',
    2: 'Yellow',
    3: 'Green',
    4: 'GreenLeft'
}
RED_LIGHT_CLASSES = ['Red', 'RedLeft', 'Yellow']

VIOLATION_ZONE = np.array([
    [1149, 471],
    [1121, 518],
    [32, 428],
    [69, 381]
], np.int32)

RECORD_SECONDS_BEFORE = 0.5
RECORD_SECONDS_AFTER = 0.5
VEHICLE_CLASS_NAMES = {
    0: 'Car',
    3: 'Motorcycle',
    1: 'Bus',
    2: 'Truck'
}