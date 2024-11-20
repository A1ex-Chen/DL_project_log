@smart_inference_mode()
def _load(self, weights: str, task: str):
    """Loads an existing NAS model weights or creates a new NAS model with pretrained weights if not provided."""
    import super_gradients
    suffix = Path(weights).suffix
    if suffix == '.pt':
        self.model = torch.load(weights)
    elif suffix == '':
        self.model = super_gradients.training.models.get(weights,
            pretrained_weights='coco')
    self.model.fuse = lambda verbose=True: self.model
    self.model.stride = torch.tensor([32])
    self.model.names = dict(enumerate(self.model._class_names))
    self.model.is_fused = lambda : False
    self.model.yaml = {}
    self.model.pt_path = weights
    self.model.task = 'detect'
