@register_to_config
def __init__(self, num_vec_classes: int, num_train_timesteps: int=100,
    alpha_cum_start: float=0.99999, alpha_cum_end: float=9e-06,
    gamma_cum_start: float=9e-06, gamma_cum_end: float=0.99999):
    self.num_embed = num_vec_classes
    self.mask_class = self.num_embed - 1
    at, att = alpha_schedules(num_train_timesteps, alpha_cum_start=
        alpha_cum_start, alpha_cum_end=alpha_cum_end)
    ct, ctt = gamma_schedules(num_train_timesteps, gamma_cum_start=
        gamma_cum_start, gamma_cum_end=gamma_cum_end)
    num_non_mask_classes = self.num_embed - 1
    bt = (1 - at - ct) / num_non_mask_classes
    btt = (1 - att - ctt) / num_non_mask_classes
    at = torch.tensor(at.astype('float64'))
    bt = torch.tensor(bt.astype('float64'))
    ct = torch.tensor(ct.astype('float64'))
    log_at = torch.log(at)
    log_bt = torch.log(bt)
    log_ct = torch.log(ct)
    att = torch.tensor(att.astype('float64'))
    btt = torch.tensor(btt.astype('float64'))
    ctt = torch.tensor(ctt.astype('float64'))
    log_cumprod_at = torch.log(att)
    log_cumprod_bt = torch.log(btt)
    log_cumprod_ct = torch.log(ctt)
    self.log_at = log_at.float()
    self.log_bt = log_bt.float()
    self.log_ct = log_ct.float()
    self.log_cumprod_at = log_cumprod_at.float()
    self.log_cumprod_bt = log_cumprod_bt.float()
    self.log_cumprod_ct = log_cumprod_ct.float()
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-
        1].copy())
