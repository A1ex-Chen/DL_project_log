def ddim_step(self, pred_x0, pred_noise, timestep_index):
    alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev,
        timestep_index, pred_x0.shape)
    dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
    x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
    return x_prev
