@torch.no_grad()
def decode_to_image(self, latents, device, size: int=64, ray_batch_size:
    int=4096, n_coarse_samples=64, n_fine_samples=128):
    projected_params = self.params_proj(latents)
    for name, param in self.mlp.state_dict().items():
        if f'nerstf.{name}' in projected_params.keys():
            param.copy_(projected_params[f'nerstf.{name}'].squeeze(0))
    camera = create_pan_cameras(size)
    rays = camera.camera_rays
    rays = rays.to(device)
    n_batches = rays.shape[1] // ray_batch_size
    coarse_sampler = StratifiedRaySampler()
    images = []
    for idx in range(n_batches):
        rays_batch = rays[:, idx * ray_batch_size:(idx + 1) * ray_batch_size]
        _, fine_sampler, coarse_model_out = self.render_rays(rays_batch,
            coarse_sampler, n_coarse_samples)
        channels, _, _ = self.render_rays(rays_batch, fine_sampler,
            n_fine_samples, prev_model_out=coarse_model_out)
        images.append(channels)
    images = torch.cat(images, dim=1)
    images = images.view(*camera.shape, camera.height, camera.width, -1
        ).squeeze(0)
    return images
