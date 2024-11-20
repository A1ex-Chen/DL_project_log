def register_lang_encoder(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _lang_encoders[model_name] = fn
    return fn
