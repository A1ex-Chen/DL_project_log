def to_numpy(self, tensor):
    return tensor.detach().cpu().numpy(
        ) if tensor.requires_grad else tensor.cpu().numpy()
