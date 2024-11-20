def _embed_masks(self, masks: torch.Tensor) ->torch.Tensor:
    """Embeds mask inputs."""
    return self.mask_downscaling(masks)
