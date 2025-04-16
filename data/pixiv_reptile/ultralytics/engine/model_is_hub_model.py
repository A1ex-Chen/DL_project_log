@staticmethod
def is_hub_model(model: str) ->bool:
    """Check if the provided model is a HUB model."""
    return any((model.startswith(f'{HUB_WEB_ROOT}/models/'), [len(x) for x in
        model.split('_')] == [42, 20], len(model) == 20 and not Path(model)
        .exists() and all(x not in model for x in './\\')))
