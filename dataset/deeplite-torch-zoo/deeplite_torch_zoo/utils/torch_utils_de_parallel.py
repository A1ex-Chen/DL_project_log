def de_parallel(model):
    return model.module if is_parallel(model) else model
