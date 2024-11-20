def __init__(self, model, device=None, points_batch_size=100000, threshold=
    0.5, refinement_step=0, resolution0=16, upsampling_steps=3,
    with_normals=False, padding=0.1, sample=False, simplify_nfaces=None,
    n_time_steps=17, only_end_time_points=False, **kwargs):
    self.n_time_steps = n_time_steps
    self.only_end_time_points = only_end_time_points
    self.onet_generator = Generator3DONet(model, device=device,
        points_batch_size=points_batch_size, threshold=threshold,
        resolution0=resolution0, upsampling_steps=upsampling_steps, padding
        =padding)
