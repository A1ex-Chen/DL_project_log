def __init__(self, model, device=None, points_batch_size=100000, threshold=
    0.5, refinement_step=0, resolution0=16, upsampling_steps=3, padding=0.1,
    sample=False, simplify_nfaces=None, n_time_steps=17, mesh_color=False,
    only_end_time_points=False, interpolate=False, fix_z=False, fix_zt=
    False, **kwargs):
    self.n_time_steps = n_time_steps
    self.mesh_color = mesh_color
    self.only_end_time_points = only_end_time_points
    self.interpolate = interpolate
    self.fix_z = fix_z
    self.fix_zt = fix_zt
    self.onet_generator = Generator3DONet(model, device=device,
        points_batch_size=points_batch_size, threshold=threshold,
        refinement_step=refinement_step, resolution0=resolution0,
        upsampling_steps=upsampling_steps, with_normals=False, padding=
        padding, sample=sample, simplify_nfaces=simplify_nfaces)
    if fix_z:
        self.fixed_z, _ = self.onet_generator.model.get_z_from_prior((1,),
            sample=sample)
    if fix_zt:
        _, self.fixed_zt = self.onet_generator.model.get_z_from_prior((1,),
            sample=sample)
