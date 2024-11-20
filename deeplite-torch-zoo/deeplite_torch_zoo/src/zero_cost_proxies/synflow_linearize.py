@torch.no_grad()
def linearize(model):
    for param in model.state_dict().values():
        param.abs_()
