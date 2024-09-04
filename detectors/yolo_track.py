from ultralytics import YOLO

class YOLO_Tracker:
    def __init__(self, path, to_tensor, device) -> None:
        self.model = YOLO(path)
        self.to_tensor = to_tensor
        self.device = device
        
    def track()