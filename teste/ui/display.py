# ui/display.py
import cv2

def draw_person_annotation(frame, box, global_id, is_stopped, elapsed, config):
    x1, y1, x2, y2 = map(int, box)
    color = config.COLOR_STOPPED if is_stopped else config.COLOR_MOVING
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID:{global_id}"
    if is_stopped:
        label += f" PARADO {int(elapsed)}s"
    
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, 1, cv2.LINE_AA)