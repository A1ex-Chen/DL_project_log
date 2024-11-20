@torch.jit.ignore
def no_weight_decay(self):
    return {'pos_embed', 'cls_token',
        'beit3.encoder.embed_positions.A.weight',
        'beit3.vision_embed.cls_token', 'logit_scale'}
