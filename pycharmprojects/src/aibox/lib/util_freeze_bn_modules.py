@staticmethod
def freeze_bn_modules(module: nn.Module):
    bn_modules = nn.ModuleList([it for it in module.modules() if isinstance
        (it, nn.BatchNorm2d)])
    for bn_module in bn_modules:
        bn_module.eval()
        for parameter in bn_module.parameters():
            parameter.requires_grad = False
