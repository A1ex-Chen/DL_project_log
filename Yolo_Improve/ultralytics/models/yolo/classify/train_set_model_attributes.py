def set_model_attributes(self):
    """Set the YOLO model's class names from the loaded dataset."""
    self.model.names = self.data['names']
