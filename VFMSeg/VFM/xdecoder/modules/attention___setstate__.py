def __setstate__(self, state):
    if '_qkv_same_embed_dim' not in state:
        state['_qkv_same_embed_dim'] = True
    super(MultiheadAttention, self).__setstate__(state)
