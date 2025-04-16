def _normal_init_(std, mean=0.0):
    return partial(torch.nn.init.normal_, mean=mean, std=std)
