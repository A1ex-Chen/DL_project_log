def dummy_model(self):

    def model(sample, t, *args):
        if isinstance(t, torch.Tensor):
            num_dims = len(sample.shape)
            t = t.reshape(-1, *((1,) * (num_dims - 1))).to(sample.device).to(
                sample.dtype)
        return sample * t / (t + 1)
    return model
