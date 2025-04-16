def _prune_heads(self, heads_to_prune):
    """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
    for layer, heads in heads_to_prune.items():
        group_idx = int(layer / self.config.inner_group_num)
        inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
        self.encoder.albert_layer_groups[group_idx].albert_layers[
            inner_group_idx].attention.prune_heads(heads)
