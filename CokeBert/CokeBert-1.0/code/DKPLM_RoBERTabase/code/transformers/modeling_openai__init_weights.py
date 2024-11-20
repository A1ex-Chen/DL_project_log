def _init_weights(self, module):
    """ Initialize the weights.
        """
    if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
