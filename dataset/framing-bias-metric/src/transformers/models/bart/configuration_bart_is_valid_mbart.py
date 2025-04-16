def is_valid_mbart(self) ->bool:
    """Is the configuration aligned with the MBART paper."""
    if (self.normalize_before and self.add_final_layer_norm and self.
        scale_embedding):
        return True
    if (self.normalize_before or self.add_final_layer_norm or self.
        scale_embedding):
        logger.info(
            'This configuration is a mixture of MBART and BART settings')
    return False
