def __init__(self, model, optimizer, dist=None, fp16=False,
    global_gradient_clip_ratio=0.0, weight_decay=0.0):
    self.model = model
    self.optimizer = optimizer
    self.dist = self.initialize_dist(dist)
    self.fp16 = fp16
    self.global_gradient_clip_ratio = global_gradient_clip_ratio
    self.weight_decay = weight_decay
