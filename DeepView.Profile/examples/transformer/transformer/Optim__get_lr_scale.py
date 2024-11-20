def _get_lr_scale(self):
    return np.min([np.power(self.n_current_steps, -0.5), np.power(self.
        n_warmup_steps, -1.5) * self.n_current_steps])
