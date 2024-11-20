@torch.jit.ignore
def no_weight_decay(self) ->Set:
    return {'pos_embed', 'cls_token', 'dist_token'}
