@torch.jit.ignore
def no_weight_decay(self):
    return {'absolute_pos_embed'}
