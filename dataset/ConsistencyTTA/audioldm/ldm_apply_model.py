def apply_model(self, x_noisy, t, cond, return_ids=False):
    if isinstance(cond, dict):
        pass
    else:
        if not isinstance(cond, list):
            cond = [cond]
        if self.model.conditioning_key == 'concat':
            key = 'c_concat'
        elif self.model.conditioning_key == 'crossattn':
            key = 'c_crossattn'
        else:
            key = 'c_film'
        cond = {key: cond}
    x_recon = self.model(x_noisy, t, **cond)
    if isinstance(x_recon, tuple) and not return_ids:
        return x_recon[0]
    else:
        return x_recon
