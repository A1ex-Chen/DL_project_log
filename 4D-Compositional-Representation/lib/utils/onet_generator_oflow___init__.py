def __init__(self, model, points_batch_size=100000, threshold=0.5,
    refinement_step=0, device=None, resolution0=16, upsampling_steps=3,
    with_normals=False, padding=0.1, sample=False, simplify_nfaces=None,
    preprocessor=None):
    self.model = model.to(device)
    self.points_batch_size = points_batch_size
    self.refinement_step = refinement_step
    self.threshold = threshold
    self.device = device
    self.resolution0 = resolution0
    self.upsampling_steps = upsampling_steps
    self.with_normals = with_normals
    self.padding = padding
    self.sample = sample
    self.simplify_nfaces = simplify_nfaces
    self.preprocessor = preprocessor
