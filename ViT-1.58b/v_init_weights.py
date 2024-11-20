def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', '']='') ->None:
    assert mode in ('jax', 'jax_nlhb', 'moco', '')
    head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
    trunc_normal_(self.pos_embed, std=0.02)
    if self.cls_token is not None:
        nn.init.normal_(self.cls_token, std=1e-06)
    named_apply(get_init_weights_vit(mode, head_bias), self)
