@torch.no_grad()
def sample(self, batch_size=16, return_intermediates=False):
    shape = batch_size, channels, self.latent_t_size, self.latent_f_size
    channels = self.channels
    return self.p_sample_loop(shape, return_intermediates=return_intermediates)
