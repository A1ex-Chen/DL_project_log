@staticmethod
def backward(ctx, *grads):
    with torch.no_grad():
        for g, grad in zip(buffer_incoming_grads, grads):
            if g is not None:
                g.copy_(grad)
    bwd_graph.replay()
    return tuple(b.detach() if b is not None else b for b in buffer_grad_inputs
        )
