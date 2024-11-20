@staticmethod
def ldm_transform_latent(z, extra_scale_factor=1):
    channel_means = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
    channel_stds = [0.9654121, 1.0440036, 0.76147926, 0.77022034]
    if len(z.shape) != 4:
        raise ValueError()
    z = z * 0.18215
    channels = [z[:, i] for i in range(z.shape[1])]
    channels = [(extra_scale_factor * (c - channel_means[i]) / channel_stds
        [i]) for i, c in enumerate(channels)]
    return torch.stack(channels, dim=1)
