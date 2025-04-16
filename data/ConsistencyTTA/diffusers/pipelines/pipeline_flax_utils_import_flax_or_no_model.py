def import_flax_or_no_model(module, class_name):
    try:
        class_obj = getattr(module, 'Flax' + class_name)
    except AttributeError:
        class_obj = getattr(module, class_name)
    except AttributeError:
        raise ValueError(
            f'Neither Flax{class_name} nor {class_name} exist in {module}')
    return class_obj
