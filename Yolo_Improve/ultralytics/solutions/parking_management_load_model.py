def load_model(self):
    """Load the Ultralytics YOLOv8 model for inference and analytics."""
    from ultralytics import YOLO
    self.model = YOLO(self.model_path)
    return self.model
