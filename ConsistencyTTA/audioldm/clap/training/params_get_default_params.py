def get_default_params(model_name):
    model_name = model_name.lower()
    if 'vit' in model_name:
        return {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-06}
    else:
        return {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08}
