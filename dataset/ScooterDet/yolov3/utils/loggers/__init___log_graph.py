def log_graph(self, model, imgsz=(640, 640)):
    if self.tb:
        log_tensorboard_graph(self.tb, model, imgsz)
