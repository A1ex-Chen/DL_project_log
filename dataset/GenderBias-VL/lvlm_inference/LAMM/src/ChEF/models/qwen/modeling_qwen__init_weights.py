def _init_weights(self, module):
    """Initialize the weights."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)
    for name, p in module.named_parameters():
        if name == 'c_proj.weight':
            p.data.normal_(mean=0.0, std=self.config.initializer_range /
                math.sqrt(2 * self.config.num_hidden_layers))
