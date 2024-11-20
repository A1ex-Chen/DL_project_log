def build_model_params(model_name, is_training, batch_norm, num_classes,
    activation, dtype, weight_decay, weight_init):
    return {'model_name': model_name, 'model_weights_path': '',
        'weights_format': 'saved_model', 'overrides': {'is_training':
        is_training, 'batch_norm': batch_norm, 'rescale_input': True,
        'num_classes': num_classes, 'weight_decay': weight_decay,
        'activation': activation, 'dtype': dtype, 'weight_init': weight_init}}
