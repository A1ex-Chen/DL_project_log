def forward(self, x, *args, **kwargs):
    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
    return self.p_losses(x, t, *args, **kwargs)
