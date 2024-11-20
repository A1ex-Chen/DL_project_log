def get_loss(model_pred, target, gt_wav, prompt, timesteps, t_indices):
    if self.snr_gamma is None:
        return self.loss(model_pred, target, gt_wav, prompt).mean()
    else:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps)
        assert len(timesteps.shape) < 2
        timesteps = timesteps.reshape(-1)
        snr = self.compute_snr(timesteps, t_indices).reshape(-1)
        mse_loss_weights = torch.clamp(snr, max=self.snr_gamma)
        instance_loss = self.loss(model_pred, target, gt_wav, prompt)
        return (instance_loss * mse_loss_weights.to(instance_loss.device)
            ).mean()
