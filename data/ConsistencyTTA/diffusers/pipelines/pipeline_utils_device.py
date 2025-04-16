@property
def device(self) ->torch.device:
    """
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
    module_names, _, _ = self.extract_init_dict(dict(self.config))
    for name in module_names.keys():
        module = getattr(self, name)
        if isinstance(module, torch.nn.Module):
            return module.device
    return torch.device('cpu')
