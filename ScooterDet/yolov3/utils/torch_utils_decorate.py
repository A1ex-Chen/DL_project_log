def decorate(fn):
    return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)
