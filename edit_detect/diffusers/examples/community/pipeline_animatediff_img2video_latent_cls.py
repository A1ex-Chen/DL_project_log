def latent_cls(v0, v1, index):
    return slerp(v0, v1, index / num_frames * (1 - strength))
