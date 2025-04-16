def final_eval(self):
    """Performs final evaluation and validation for object detection YOLO-World model."""
    val = self.args.data['val']['yolo_data'][0]
    self.validator.args.data = val
    self.validator.args.split = 'minival' if isinstance(val, str
        ) and 'lvis' in val else 'val'
    return super().final_eval()
