@staticmethod
def backward(ctx, *grads):
    all_gradients = torch.stack(grads)
    torch.distributed.all_reduce(all_gradients)
    return all_gradients[torch.distributed.get_rank()]
