def prune_heads(self, heads_to_prune):
    """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
    for layer, heads in heads_to_prune.items():
        union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
        self.config.pruned_heads[layer] = list(union_heads)
    self.base_model._prune_heads(heads_to_prune)