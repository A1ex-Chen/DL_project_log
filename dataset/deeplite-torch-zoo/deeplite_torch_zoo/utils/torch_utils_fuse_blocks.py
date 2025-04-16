def fuse_blocks(model: torch.nn.Module) ->nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'fuse'):
            module.fuse()
    return model
