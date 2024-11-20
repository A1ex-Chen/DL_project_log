def get_loss(model_pred, target, timesteps):
    if self.snr_gamma is None:
        return F.mse_loss(model_pred.float(), target.float(), reduction='mean')
    else:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps)
        assert len(timesteps.shape) < 2
        timesteps = timesteps.reshape(-1)
        snr = self.compute_snr(timesteps).reshape(-1)
        truncated_snr = torch.clamp(snr, max=self.snr_gamma)
        if self.noise_scheduler.config.prediction_type == 'v_prediction':
            mse_loss_weights = truncated_snr / (snr + 1)
        elif self.noise_scheduler.config.prediction_type == 'epsilon':
            mse_loss_weights = truncated_snr / snr
        else:
            raise ValueError('Unknown prediction type.')
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
        instance_loss = loss.mean(dim=list(range(1, len(loss.shape))))
        return (instance_loss * mse_loss_weights.to(loss.device)).mean()
