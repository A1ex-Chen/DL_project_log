@torch.jit.ignore
def no_weight_decay(self):
    return {'positional_embedding', 'token_embedding'}
