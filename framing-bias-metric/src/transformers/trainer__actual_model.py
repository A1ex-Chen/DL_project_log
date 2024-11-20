@staticmethod
def _actual_model(model: Union[torch.nn.DataParallel, torch.nn.parallel.
    DistributedDataParallel, torch.nn.modules.Module]
    ) ->torch.nn.modules.Module:
    """

        Args:
            model: (:obj:`Union[torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, torch.nn.modules.Module]`):
                Model object used during training

        Returns:
            :obj:`torch.nn.modules.Module`: unwrapped module
        """
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.
        nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model
    return model
