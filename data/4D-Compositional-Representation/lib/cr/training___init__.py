def __init__(self, model, optimizer, device=None, input_type='img',
    threshold=0.4):
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.input_type = input_type
    self.threshold = threshold
