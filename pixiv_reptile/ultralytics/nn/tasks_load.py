def load(self, weights, verbose=True):
    """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
    model = weights['model'] if isinstance(weights, dict) else weights
    csd = model.float().state_dict()
    csd = intersect_dicts(csd, self.state_dict())
    self.load_state_dict(csd, strict=False)
    if verbose:
        LOGGER.info(
            f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights'
            )
