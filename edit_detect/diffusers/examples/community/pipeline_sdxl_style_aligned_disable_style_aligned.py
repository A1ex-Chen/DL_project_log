def disable_style_aligned(self):
    """Disables the StyleAligned mechanism if it had been previously enabled."""
    if self.style_aligned_enabled:
        for layer in self._style_aligned_norm_layers:
            layer.forward = layer.orig_forward
        self._style_aligned_norm_layers = None
        self._disable_shared_attention_processors()
