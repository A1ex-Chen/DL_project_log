@torch.jit.ignore
def no_weight_decay(self):
    return {'pos_embed'}
