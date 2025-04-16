def _prune_heads(self, heads_to_prune):
    """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
    raise NotImplementedError
