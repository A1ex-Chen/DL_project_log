def get_head_mask(self, head_mask):
    if head_mask is not None:
        raise NotImplementedError
    else:
        head_mask = [None] * self.config.num_hidden_layers
    return head_mask
