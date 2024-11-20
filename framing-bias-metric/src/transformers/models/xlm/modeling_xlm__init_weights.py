def _init_weights(self, module):
    """ Initialize the weights. """
    if isinstance(module, nn.Embedding):
        if self.config is not None and self.config.embed_init_std is not None:
            nn.init.normal_(module.weight, mean=0, std=self.config.
                embed_init_std)
    if isinstance(module, nn.Linear):
        if self.config is not None and self.config.init_std is not None:
            nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
