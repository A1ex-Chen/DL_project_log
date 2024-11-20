@classmethod
def convert_frozen_batchnorm(cls, module):
    """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
    bn_module = nn.modules.batchnorm
    bn_module = bn_module.BatchNorm2d, bn_module.SyncBatchNorm
    res = module
    if isinstance(module, bn_module):
        res = cls(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = cls.convert_frozen_batchnorm(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res
