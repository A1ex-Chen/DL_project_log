@torch.no_grad()
def __call__(self, features: torch.Tensor, schedule=[1.0, 0.5], generator=None
    ):
    features = self.ldm_transform_latent(features)
    ts = self.round_timesteps(torch.arange(0, 1024), 1024, self.
        n_distilled_steps, truncate_start=False)
    shape = features.size(0), 3, 8 * features.size(2), 8 * features.size(3)
    x_start = torch.zeros(shape, device=features.device, dtype=features.dtype)
    schedule_timesteps = [int((1024 - 1) * s) for s in schedule]
    for i in schedule_timesteps:
        t = ts[i].item()
        t_ = torch.tensor([t] * features.shape[0]).to(self.device)
        noise = torch.randn(x_start.shape, dtype=x_start.dtype, generator=
            generator).to(device=x_start.device)
        x_start = _extract_into_tensor(self.sqrt_alphas_cumprod, t_,
            x_start.shape) * x_start + _extract_into_tensor(self.
            sqrt_one_minus_alphas_cumprod, t_, x_start.shape) * noise
        c_in = _extract_into_tensor(self.c_in, t_, x_start.shape)
        import torch.nn.functional as F
        from diffusers import UNet2DModel
        if isinstance(self.ckpt, UNet2DModel):
            input = torch.concat([c_in * x_start, F.upsample_nearest(
                features, scale_factor=8)], dim=1)
            model_output = self.ckpt(input, t_).sample
        else:
            model_output = self.ckpt(c_in * x_start, t_, features=features)
        B, C = x_start.shape[:2]
        model_output, _ = torch.split(model_output, C, dim=1)
        pred_xstart = (_extract_into_tensor(self.c_out, t_, x_start.shape) *
            model_output + _extract_into_tensor(self.c_skip, t_, x_start.
            shape) * x_start).clamp(-1, 1)
        x_start = pred_xstart
    return x_start
