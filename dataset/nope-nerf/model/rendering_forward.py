def forward(self, pixels, depth, camera_mat, world_mat, scale_mat,
    rendering_technique, add_noise=True, eval_=False, it=1000000):
    if rendering_technique == 'nope_nerf':
        out_dict = self.nope_nerf(pixels, depth, camera_mat, world_mat,
            scale_mat, it=it, add_noise=add_noise, eval_=eval_)
    elif rendering_technique == 'phong_renderer':
        out_dict = self.phong_renderer(pixels, camera_mat, world_mat,
            scale_mat, it=it)
    return out_dict
