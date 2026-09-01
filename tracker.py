import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


class YOLODetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45,
                 tracker_path: str = "bytetrack.yaml", vehicle_classes=None):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_path = tracker_path
        self.vehicle_classes = vehicle_classes  
        self.class_names = self.model.names    
    def reset_tracker(self):
        
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers[0].reset()

    def track_and_detect(self, frame: np.ndarray, img_size: int = 640):
        results = self.model.track(
            source=frame,
            persist=True,
            imgsz=img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_path,
            verbose=False
        )

        result = results[0]
        tracked_objects = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                class_id = int(class_id)
                if self.vehicle_classes is not None and class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box
                tracked_objects.append({
                    "id": int(track_id),
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "class_id": class_id,
                    "confidence": float(conf)
                })

        return tracked_objects