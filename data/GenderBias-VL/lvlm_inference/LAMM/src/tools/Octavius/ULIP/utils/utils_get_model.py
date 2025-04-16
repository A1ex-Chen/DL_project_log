def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.
        nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
