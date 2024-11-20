@property
def device(self) ->torch.device:
    """
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
    module_names, _ = self._get_signature_keys(self)
    modules = [getattr(self, n, None) for n in module_names]
    modules = [m for m in modules if isinstance(m, torch.nn.Module)]
    for module in modules:
        return module.device
    return torch.device('cpu')
