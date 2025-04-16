def init_weights(self):
    """ Initialize and prunes weights if needed. """
    self.apply(self._init_weights)
    if self.config.pruned_heads:
        self.prune_heads(self.config.pruned_heads)
    self.tie_weights()
