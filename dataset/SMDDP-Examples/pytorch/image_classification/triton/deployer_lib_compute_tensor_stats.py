def compute_tensor_stats(self, tensor):
    return {'std': tensor.std().item(), 'mean': tensor.mean().item(), 'max':
        tensor.max().item(), 'min': tensor.min().item()}
