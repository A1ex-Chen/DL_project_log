@staticmethod
def backward(ctx, *grads):
    all_gradients = torch.stack(grads)
    dist.all_reduce(all_gradients)
    return all_gradients[dist.get_rank()]
