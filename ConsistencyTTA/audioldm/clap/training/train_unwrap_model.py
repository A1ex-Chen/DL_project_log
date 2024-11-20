def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model
