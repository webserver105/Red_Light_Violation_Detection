from utils import is_center_in_zone
import config

def check_violations(tracked_vehicles, traffic_light_status, zone_polygon, violation_ids):
    newly_detected_violators = []

    is_light_red = traffic_light_status in config.RED_LIGHT_CLASSES
    if not is_light_red:
        return violation_ids, newly_detected_violators
    
    for track in tracked_vehicles:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        
        if track_id in violation_ids:
            continue

        bbox = track.to_tlbr()
        box_center_bottom = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))

        if is_center_in_zone(box_center_bottom, zone_polygon):
            print(f"🚨 VIOLATION DETECTED: Vehicle ID {track_id} entered zone during red light!")
            violation_ids.add(track_id)
            newly_detected_violators.append(track)
            
    return violation_ids, newly_detected_violators