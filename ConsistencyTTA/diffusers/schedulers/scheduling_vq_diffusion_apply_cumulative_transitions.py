def apply_cumulative_transitions(self, q, t):
    bsz = q.shape[0]
    a = self.log_cumprod_at[t]
    b = self.log_cumprod_bt[t]
    c = self.log_cumprod_ct[t]
    num_latent_pixels = q.shape[2]
    c = c.expand(bsz, 1, num_latent_pixels)
    q = (q + a).logaddexp(b)
    q = torch.cat((q, c), dim=1)
    return q
