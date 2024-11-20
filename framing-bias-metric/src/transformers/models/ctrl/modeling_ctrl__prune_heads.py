def _prune_heads(self, heads_to_prune):
    """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
    for layer, heads in heads_to_prune.items():
        self.h[layer].multi_head_attention.prune_heads(heads)
