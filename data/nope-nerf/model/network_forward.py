def forward(self, p, ray_idx, camera_mat, world_mat, scale_mat,
    rendering_technique, it=0, eval_mode=False, depth_img=None, add_noise=
    True, img_size=None):
    if rendering_technique == 'nope_nerf':
        depth_img_resized = F.interpolate(depth_img, img_size, mode='nearest')
        depth_img_resized = depth_img_resized.view(1, 1, -1).permute(0, 2, 1)
        depth = depth_img_resized[:, ray_idx]
    else:
        depth = None
    out_dict = self.renderer(p, depth, camera_mat, world_mat, scale_mat,
        rendering_technique, eval_=eval_mode, it=it, add_noise=add_noise)
    return out_dict
