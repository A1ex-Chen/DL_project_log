def run_diffusion(self, x, conditions, n_guide_steps, scale):
    batch_size = x.shape[0]
    y = None
    for i in tqdm.tqdm(self.scheduler.timesteps):
        timesteps = torch.full((batch_size,), i, device=self.unet.device,
            dtype=torch.long)
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                x.requires_grad_()
                y = self.value_function(x.permute(0, 2, 1), timesteps).sample
                grad = torch.autograd.grad([y.sum()], [x])[0]
                posterior_variance = self.scheduler._get_variance(i)
                model_std = torch.exp(0.5 * posterior_variance)
                grad = model_std * grad
            grad[timesteps < 2] = 0
            x = x.detach()
            x = x + scale * grad
            x = self.reset_x0(x, conditions, self.action_dim)
        prev_x = self.unet(x.permute(0, 2, 1), timesteps).sample.permute(0,
            2, 1)
        x = self.scheduler.step(prev_x, i, x, predict_epsilon=False)[
            'prev_sample']
        x = self.reset_x0(x, conditions, self.action_dim)
        x = self.to_torch(x)
    return x, y
