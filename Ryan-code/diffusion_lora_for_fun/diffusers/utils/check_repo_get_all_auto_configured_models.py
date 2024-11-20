def get_all_auto_configured_models():
    """Return the list of all models in at least one auto class."""
    result = set()
    if is_torch_available():
        for attr_name in dir(diffusers.models.auto.modeling_auto):
            if attr_name.startswith('MODEL_') and attr_name.endswith(
                'MAPPING_NAMES'):
                result = result | set(get_values(getattr(diffusers.models.
                    auto.modeling_auto, attr_name)))
    if is_flax_available():
        for attr_name in dir(diffusers.models.auto.modeling_flax_auto):
            if attr_name.startswith('FLAX_MODEL_') and attr_name.endswith(
                'MAPPING_NAMES'):
                result = result | set(get_values(getattr(diffusers.models.
                    auto.modeling_flax_auto, attr_name)))
    return list(result)
