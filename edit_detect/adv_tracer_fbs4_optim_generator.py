@staticmethod
def optim_generator(name: str, lr: float, **kwargs):
    if name == Optimizer.OPTIM_ADAM:
        opt = partial(torch.optim.Adam, lr=lr, **kwargs)
    else:
        raise ValueError(f'')
    return opt
