def _embed_init(self, initrange=0.05):
    """Initialize the embedding weights"""
    nn.init.uniform_(self.embed.weight, -initrange, initrange)
