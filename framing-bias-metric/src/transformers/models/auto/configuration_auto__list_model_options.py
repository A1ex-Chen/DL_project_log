def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError(
            'Using `use_model_types=False` requires a `config_to_class` dictionary.'
            )
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: config.__name__ for 
                model_type, config in CONFIG_MAPPING.items()}
        else:
            model_type_to_name = {model_type: config_to_class[config].
                __name__ for model_type, config in CONFIG_MAPPING.items() if
                config in config_to_class}
        lines = [
            f'{indent}- **{model_type}** -- :class:`~transformers.{cls_name}` ({MODEL_NAMES_MAPPING[model_type]} model)'
             for model_type, cls_name in model_type_to_name.items()]
    else:
        config_to_name = {config.__name__: clas.__name__ for config, clas in
            config_to_class.items()}
        config_to_model_name = {config.__name__: MODEL_NAMES_MAPPING[
            model_type] for model_type, config in CONFIG_MAPPING.items()}
        lines = [
            f'{indent}- :class:`~transformers.{config_name}` configuration class: :class:`~transformers.{cls_name}` ({config_to_model_name[config_name]} model)'
             for config_name, cls_name in config_to_name.items()]
    return '\n'.join(lines)
